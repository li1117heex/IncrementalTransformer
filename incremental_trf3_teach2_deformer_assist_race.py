import torch

from transformers.models.roberta.modeling_roberta import (
    RobertaModel,
    RobertaEncoder,
    RobertaLayer,
    RobertaForQuestionAnswering,
    RobertaForMultipleChoice,
    RobertaPooler
)

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, QuestionAnsweringModelOutput, BaseModelOutputWithPastAndCrossAttentions, MultipleChoiceModelOutput
from transformers.modeling_utils import apply_chunking_to_forward
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, CosineEmbeddingLoss
from transformers import AutoConfig

from incremental_attn_p2q import IncrementalRobertaSelfAttentionCross
import copy

class IncrementalRobertaLayer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        layer_number=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # if layer_number<7:
        #     hidden_states=encoder_attention_mask*hidden_states+(1-encoder_attention_mask)*encoder_hidden_states[layer_number]
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states[layer_number],
            encoder_attention_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

class IncrementalRobertaEncoder(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        # self.config = config
        self.layer = nn.ModuleList([IncrementalRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        # self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    # logger.warning(
                    #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    # )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    layer_number=i,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class IncrementalRobertaModel(RobertaModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config, add_pooling_layer=False)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.config = config
        #
        # self.embeddings = RobertaEmbeddings(config)
        # self.encoder = RobertaEncoder(config)
        #
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        config2 = copy.deepcopy(config)
        config2.num_hidden_layers = config.encoder2_layers
        config2assist = copy.deepcopy(config)
        config2assist.num_hidden_layers = config.assist_layers
        config3 = copy.deepcopy(config)
        config3.num_hidden_layers = config.encoder3_layers
        self.encoder2 = RobertaEncoder(config2)
        self.encoder2assist = IncrementalRobertaEncoder(config2assist)
        self.encoder3 = RobertaEncoder(config3)
        for i, layer in enumerate(self.encoder2assist.layer):
            layer.attention.self = IncrementalRobertaSelfAttentionCross(config)

        # Initialize weights and apply final processing
        self.post_init()

    # def get_input_embeddings(self):
    #     return self.embeddings.word_embeddings
    #
    # def set_input_embeddings(self, value):
    #     self.embeddings.word_embeddings = value
    #
    # def _prune_heads(self, heads_to_prune):
    #     """
    #     Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
    #     class PreTrainedModel
    #     """
    #     for layer, heads in heads_to_prune.items():
    #         self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPoolingAndCrossAttentions,
    #     config_class=_CONFIG_FOR_DOC,
    # )

    def get_extended_attention_mask2(self, attention_mask, input_shape, device) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        #change
        extended_attention_mask=extended_attention_mask.repeat(1,1,input_shape[1],1)
        mask2=(extended_attention_mask.transpose(2,3)+extended_attention_mask)>2
        extended_attention_mask=mask2.type_as(extended_attention_mask)

        # mask2 = attention_mask[:, None, :, None].repeat(1, 1, 1, input_shape[1]) != 2
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def bound(self,input_ids,attention_mask):
        boundry = (attention_mask == 1).type(torch.int).sum(-1).max().item()
        input_ids=input_ids[:,:boundry]
        attention_mask=attention_mask[:,:boundry]

        return boundry,input_ids,attention_mask

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids=None,
        question_input_ids=None,
        attention_mask=None,
        question_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        # boundry,question_input_ids,question_attention_mask=self.bound(question_input_ids,question_attention_mask)
        boundry=question_input_ids.shape[-1]
        input_ids=torch.cat([question_input_ids,input_ids],dim=1)

        # attention_mask2 = torch.cat([question_attention_mask*2, attention_mask], dim=1)
        attention_mask = torch.cat([question_attention_mask, attention_mask], dim=1)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # cc=(input_ids==2).nonzero()[1::3,1]
        # for ii in range(input_ids.shape[0]):
        #     attention_mask[:cc[ii]] = torch.tensor([2]*(cc[ii]))

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # attention_mask1=(attention_mask==1).type_as(attention_mask)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # extended_attention_mask2: torch.Tensor = self.get_extended_attention_mask2(attention_mask2, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output[:,boundry:],
            attention_mask=extended_attention_mask[:,:,:,boundry:],
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        encoder2_input = embedding_output[:, :boundry]
        encoder2_output = self.encoder2(
            encoder2_input,
            attention_mask=extended_attention_mask[:,:,:,:boundry],
            # head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder1_highlayers = encoder_outputs.hidden_states[self.encoder2.config.num_hidden_layers:self.encoder2.config.num_hidden_layers + self.encoder2assist.config.num_hidden_layers]
        # encoder2_mask = extended_attention_mask2[:, :, :boundry]
        encoder2_output = self.encoder2assist(
            encoder2_output[0],
            attention_mask=extended_attention_mask,
            # head_mask=head_mask,
            encoder_hidden_states=encoder1_highlayers,
            encoder_attention_mask=extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder3_input = torch.cat([encoder2_output[0], encoder_outputs.hidden_states[self.encoder2.config.num_hidden_layers + self.encoder2assist.config.num_hidden_layers]],dim=1)

        # sequence_mask = (attention_mask != 2).type_as(encoder_outputs[0])[:, :, None]
        # encoder2_input=sequence_mask*encoder_outputs[0]+(1-sequence_mask)*embedding_output
        sequence_output = self.encoder3(
            encoder3_input,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # encoder_outputs2 = self.encoder(
        #     embedding_output,
        #     attention_mask=extended_attention_mask2,
        #     head_mask=head_mask,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=encoder_extended_attention_mask,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        # sequence_mask=(attention_mask!=2).type_as(encoder_outputs[0])[:,:,None]
        # sequence_output = encoder_outputs[0]*sequence_mask+encoder_outputs2[0]
        # sequence_output = self.fusion(encoder_outputs[0],encoder_outputs2[0],sequence_mask)
        pooled_output = self.pooler(sequence_output[0]) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output[0], pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output[0],#torch.concat([sequence_output[0][:,:boundry],torch.zeros([batch_size,50-boundry,self.config.hidden_size],dtype=torch.long, device=device),sequence_output[0][:,boundry:]],dim=1), #[0][:,boundry:],
            pooler_output=pooled_output,
            past_key_values=sequence_output.past_key_values,
            hidden_states=sequence_output.hidden_states,
            attentions=sequence_output.attentions,
            cross_attentions=sequence_output.cross_attentions,
        )

class IncrementalRobertaForMultipleChoice(RobertaForMultipleChoice):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = IncrementalRobertaModel(config, add_pooling_layer=True)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, 1)

        teacher_config = AutoConfig.from_pretrained(
            config.teacher_model_path,
            cache_dir=None,
            revision='main',
            use_auth_token=False,
        )
        self.teacher = RobertaForMultipleChoice.from_pretrained(
            config.teacher_model_path,
            from_tf=bool(".ckpt" in config.teacher_model_path),
            config=teacher_config,
            cache_dir=None,
            revision='main',
            use_auth_token=False,
        )
        # self.teacher.training=False
        self.alpha = config.alpha
        self.temperature = config.temperature

        # Initialize weights and apply final processing
        # self.post_init()

    # @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=MultipleChoiceModelOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        question_input_ids=None,
        attention_mask=None,
        question_attention_mask=None,
        token_type_ids=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        flat_question_input_ids = question_input_ids.view(-1, question_input_ids.size(-1)) if question_input_ids is not None else None
        flat_question_attention_mask = question_attention_mask.view(-1, question_attention_mask.size(-1)) if question_attention_mask is not None else None

        outputs = self.roberta(
            flat_input_ids,
            question_input_ids=flat_question_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            question_attention_mask=flat_question_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            input_ids = torch.cat([question_input_ids, input_ids], dim=2)
            attention_mask = torch.cat([question_attention_mask, attention_mask], dim=2)
            teacher_results = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # teacher_logits = teacher_results['logits'].softmax(dim=1) / self.temperature
            # sl_loss_fct = CrossEntropyLoss()
            # teach_loss = sl_loss_fct(reshaped_logits / self.temperature, teacher_logits)
            teacher_hidden_states=teacher_results['hidden_states'][-1]
            teach_loss_fct = nn.MSELoss()
            teach_loss=teach_loss_fct(teacher_hidden_states,outputs['last_hidden_state'])
            # teach_loss_fct = CosineEmbeddingLoss(reduction='mean')
            # xx1 = torch.flatten(teacher_hidden_states, start_dim=1)
            # xx2 = torch.flatten(outputs['last_hidden_state'], start_dim=1)
            # yy = torch.ones(xx1.shape[0], dtype=torch.int, device=xx1.device)
            # teach_loss = teach_loss_fct(xx1, xx2, yy)
            loss = (1 - self.alpha) * loss + self.alpha * teach_loss

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def convert_ro_incr(model,config):
    enc2_layers = model.roberta.encoder2.config.num_hidden_layers
    assist_layers=model.roberta.encoder2assist.config.num_hidden_layers
    for i in range(model.roberta.encoder2.config.num_hidden_layers):
        model.roberta.encoder2.layer[i] = model.roberta.encoder.layer[i]
    for i in range(model.roberta.encoder2assist.config.num_hidden_layers):
        model.roberta.encoder2assist.layer[i].intermediate = copy.deepcopy(
            model.roberta.encoder.layer[i + enc2_layers].intermediate)
        model.roberta.encoder2assist.layer[i].output = copy.deepcopy(model.roberta.encoder.layer[i + enc2_layers].output)
        model.roberta.encoder2assist.layer[i].attention.output = copy.deepcopy(
            model.roberta.encoder.layer[i + enc2_layers].attention.output)
        model.roberta.encoder2assist.layer[i].attention.self.query = copy.deepcopy(
            model.roberta.encoder.layer[i + enc2_layers].attention.self.query)
        model.roberta.encoder2assist.layer[i].attention.self.key = copy.deepcopy(
            model.roberta.encoder.layer[i + enc2_layers].attention.self.key)
        model.roberta.encoder2assist.layer[i].attention.self.value = copy.deepcopy(
            model.roberta.encoder.layer[i + enc2_layers].attention.self.value)
        # model.roberta.encoder2.layer[i].attention.self.query1 = copy.deepcopy(model.roberta.encoder.layer[i].attention.self.query)
        # model.roberta.encoder2.layer[i].attention.self.key1 = copy.deepcopy(model.roberta.encoder.layer[i].attention.self.key)
        # model.roberta.encoder2.layer[i].attention.self.value1 = copy.deepcopy(model.roberta.encoder.layer[i].attention.self.value)
        # model.roberta.encoder2.layer[i].attention.self.query2 = copy.deepcopy(model.roberta.encoder.layer[i].attention.self.query)
        # model.roberta.encoder2.layer[i].attention.self.key2 = copy.deepcopy(model.roberta.encoder.layer[i].attention.self.key)
        # model.roberta.encoder2.layer[i].attention.self.value2 = copy.deepcopy(model.roberta.encoder.layer[i].attention.self.value)
    for i in range(model.roberta.encoder3.config.num_hidden_layers):
        model.roberta.encoder3.layer[i] = copy.deepcopy(model.roberta.encoder.layer[i + enc2_layers+assist_layers])
    teacher_config = AutoConfig.from_pretrained(
        config.teacher_model_path,
        cache_dir=None,
        revision='main',
        use_auth_token=False,
    )
    model.teacher = RobertaForMultipleChoice.from_pretrained(
        config.teacher_model_path,
        from_tf=bool(".ckpt" in config.teacher_model_path),
        config=teacher_config,
        cache_dir=None,
        revision='main',
        use_auth_token=False,
    )
    for param in model.teacher.parameters():
        param.requires_grad = False
