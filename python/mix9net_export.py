import torch
import numpy as np
import io

from .mix9net import Mix9Net
from .line_encoding import get_total_num_encoding, get_encoding_usage_flags, transform_lines_to_line_encoding


def generate_base3_permutation(length):
    x = np.zeros((1, length), dtype=np.int8)
    for i in range(length):
        x1, x2 = x.copy(), x.copy()
        x1[:, i] = 1
        x2[:, i] = 2
        x = np.concatenate((x, x1, x2), axis=0)
    return x


class Mix9NetSerializer():
    """
    Mix9Net binary serializer.

    The corresponding C++ language struct layout: 
    struct Mix9Weight {
        // 1  mapping layer
        int16_t mapping[2][ShapeNum][FeatureDim];

        // 2  Depthwise conv
        int16_t feature_dwconv_weight[9][FeatDWConvDim];
        int16_t feature_dwconv_bias[FeatDWConvDim];

        struct HeadBucket {
            // 3  Policy dynamic pointwise conv
            int8_t  policy_pwconv_layer_l1_weight[(PolicyDim * 2) * FeatureDim];
            int32_t policy_pwconv_layer_l1_bias[PolicyDim * 2];
            int8_t  policy_pwconv_layer_l2_weight[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim) * (PolicyDim * 2)];
            int32_t policy_pwconv_layer_l2_bias[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim)];
            
            // 4  Value Group MLP (layer 1,2)
            int8_t  value_corner_weight[ValueDim * FeatureDim];
            int32_t value_corner_bias[ValueDim];
            int8_t  value_edge_weight[ValueDim * FeatureDim];
            int32_t value_edge_bias[ValueDim];
            int8_t  value_center_weight[ValueDim * FeatureDim];
            int32_t value_center_bias[ValueDim];
            int8_t  value_quad_weight[ValueDim * ValueDim];
            int32_t value_quad_bias[ValueDim];

            // 5  Value MLP (layer 1,2,3)
            int8_t  value_l1_weight[ValueDim * (FeatureDim + ValueDim * 4)];
            int32_t value_l1_bias[ValueDim];
            int8_t  value_l2_weight[ValueDim * ValueDim];
            int32_t value_l2_bias[ValueDim];
            int8_t  value_l3_weight[4 * ValueDim];
            int32_t value_l3_bias[4];

            // 6  Policy output linear
            float policy_output_weight[16];
            float policy_output_bias;
            char  __padding_to_64bytes_1[44];
        } buckets[NumHeadBucket];
    };
    """

    def __init__(self, text_output=False):
        self.line_length = 11
        self.text_output = text_output
        self.map_table_export_batch_size = 4096

    def _export_map_table(self, model: Mix9Net, device, line, mapping_idx):
        """
        Export line -> feature mapping table.

        Args:
            line: shape (N, Length)
        """
        N, L = line.shape
        us, opponent = line == 1, line == 2
        line = np.stack((us, opponent), axis=0)[np.newaxis]  # [1, C=2, N, L]
        line = torch.tensor(line, dtype=torch.float32, device=device)

        batch_size = self.map_table_export_batch_size
        batch_num = 1 + (N - 1) // batch_size
        map_table = []
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)

            mapping = getattr(model, f'mapping{mapping_idx}')
            feature = mapping(line[:, :, start:end],
                              dirs=[0])[0, 0]  # [dim_feature, batch, L]
            feature = torch.permute(
                feature, (1, 2, 0)).cpu().numpy()  # [batch, L, dim_feature]
            map_table.append(feature)

        map_table = np.concatenate(map_table, axis=0)  # [N, L, dim_feature]
        return map_table

    def _export_feature_map(self, model: Mix9Net, device, mapping_idx):
        # use line encoding to generate feature map
        L = self.line_length
        dim_feature = model.model_size[1]
        num_encoding = get_total_num_encoding(L)
        usage_flags = get_encoding_usage_flags(L)  # [N] bool, track usage of each feature
        feature_map = np.zeros((num_encoding, dim_feature),
                               dtype=np.float32)  # [N, dim_feature]

        # generate line features
        for l in reversed(range(1, L + 1)):
            lines = generate_base3_permutation(l)
            idxs = transform_lines_to_line_encoding(lines, L)  # [n, 11]
            map_table = self._export_map_table(model, device, lines,
                                               mapping_idx)
            rows = np.arange(idxs.shape[0])[:, None]  # [n, 1]
            feature_map[idxs, :] = map_table[rows, np.arange(l), :]

        feature_map_clipped = np.clip(feature_map, a_min=-16, a_max=511 / 32)
        num_params_clipped = np.sum(feature_map != feature_map_clipped)
        feature_map_quant = np.around(feature_map_clipped * 32).astype(np.int16)  # [-512, 511]
        print(
            f"feature map: used {usage_flags.sum()} features of {len(usage_flags)}, "
            + f": clipped {num_params_clipped}/{feature_map.size}" +
            f", quant_range = {(feature_map_quant.min(), feature_map_quant.max())}"
        )
        return feature_map_quant, usage_flags

    def _export_feature_dwconv(self, model: Mix9Net):
        conv_weight = model.feature_dwconv.conv.weight.cpu().numpy()  # [FeatureDWConvDim,1,3,3]
        conv_bias = model.feature_dwconv.conv.bias.cpu().numpy()  # [FeatureDWConvDim]
        conv_weight = conv_weight.reshape(conv_weight.shape[0], -1).transpose()  # [9, FeatureDWConvDim]

        conv_weight_clipped = np.clip(conv_weight, a_min=-32768 / 65536, a_max=32767 / 65536)
        conv_bias_clipped = np.clip(conv_bias, a_min=-64, a_max=64)  # not too large, otherwise it may overflow
        weight_num_params_clipped = np.sum(conv_weight != conv_weight_clipped)
        bias_num_params_clipped = np.sum(conv_bias != conv_bias_clipped)
        conv_weight_quant = np.clip(np.around(conv_weight_clipped * 65536),
                                    -32768, 32767).astype(np.int16)
        conv_bias_quant = np.clip(np.around(conv_bias_clipped * 128), -32768,
                                  32767).astype(np.int16)
        print(
            f"feature dwconv: weight clipped {weight_num_params_clipped}/{conv_weight.size}"
            + f", bias clipped {bias_num_params_clipped}/{conv_bias.size}" +
            f", weight_quant_range = {(conv_weight_quant.min(), conv_weight_quant.max())}"
            +
            f", bias_quant_range = {(conv_bias_quant.min(), conv_bias_quant.max())}"
        )

        # Make sure that the dwconv will not overflow
        assert np.all(np.abs(conv_weight_clipped).sum(0)/2*16*4*128 < 32767), \
            f"feature dwconv would overflow! (maxsum={np.abs(conv_weight_clipped).sum(0).max()})"

        return conv_weight_quant, conv_bias_quant

    def _export_policy_pwconv(self, model: Mix9Net):
        # policy pw conv dynamic weight layer 1
        l1_weight = model.policy_pwconv_weight_linear[0].fc.weight.cpu().numpy()
        l1_bias = model.policy_pwconv_weight_linear[0].fc.bias.cpu().numpy()

        # policy pw conv dynamic weight layer 2
        l2_weight = model.policy_pwconv_weight_linear[1].fc.weight.cpu().numpy()
        l2_bias = model.policy_pwconv_weight_linear[1].fc.bias.cpu().numpy()

        # policy PReLU activation
        policy_output_weight = model.policy_output.weight.cpu().squeeze().numpy()
        policy_output_bias = model.policy_output.bias.cpu().numpy()

        return (
            np.clip(np.around(l1_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * 128 * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * 128 * 128), -2**31, 2**31 - 1).astype(np.int32),
            policy_output_weight / (128 * 128 * 128),
            policy_output_bias,
        )

    def _export_value(self, model: Mix9Net):
        # group layers
        corner_weight = model.value_corner_linear.fc.weight.cpu().numpy()
        corner_bias = model.value_corner_linear.fc.bias.cpu().numpy()
        edge_weight = model.value_edge_linear.fc.weight.cpu().numpy()
        edge_bias = model.value_edge_linear.fc.bias.cpu().numpy()
        center_weight = model.value_center_linear.fc.weight.cpu().numpy()
        center_bias = model.value_center_linear.fc.bias.cpu().numpy()
        quad_weight = model.value_quad_linear.fc.weight.cpu().numpy()
        quad_bias = model.value_quad_linear.fc.bias.cpu().numpy()

        # value layers
        l1_weight = model.value_linear[0].fc.weight.cpu().numpy()
        l1_bias = model.value_linear[0].fc.bias.cpu().numpy()
        l2_weight = model.value_linear[1].fc.weight.cpu().numpy()
        l2_bias = model.value_linear[1].fc.bias.cpu().numpy()
        l3_weight = model.value_linear[2].fc.weight.cpu().numpy()
        l3_bias = model.value_linear[2].fc.bias.cpu().numpy()

        return (
            np.clip(np.around(corner_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(corner_bias * 128 * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(edge_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(edge_bias * 128 * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(center_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(center_bias * 128 * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(quad_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(quad_bias * 128 * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l1_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l1_bias * 128 * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l2_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l2_bias * 128 * 128), -2**31, 2**31 - 1).astype(np.int32),
            np.clip(np.around(l3_weight * 128), -128, 127).astype(np.int8),
            np.clip(np.around(l3_bias * 128 * 128), -2**31, 2**31 - 1).astype(np.int32),
        )

    def serialize(self, out: io.IOBase, model: Mix9Net, device):
        feature_map1, usage_flags1 = self._export_feature_map(model, device, 1)
        feature_map2, usage_flags2 = self._export_feature_map(model, device, 2)
        feat_dwconv_weight, feat_dwconv_bias = self._export_feature_dwconv(model)
        policy_pwconv_layer_l1_weight, policy_pwconv_layer_l1_bias, \
            policy_pwconv_layer_l2_weight, policy_pwconv_layer_l2_bias, \
            policy_output_weight, policy_output_bias = self._export_policy_pwconv(model)
        corner_weights, corner_bias, edge_weights, edge_bias, \
            center_weights, center_bias, quad_weights, quad_bias, \
            l1_weight, l1_bias, l2_weight, l2_bias, l3_weight, l3_bias = self._export_value(model)

        if self.text_output:
            print('featuremap1', file=out)
            print(usage_flags1.sum(), file=out)
            for i, (f, used) in enumerate(
                    zip(feature_map1.astype('i2'), usage_flags1)):
                if used:
                    print(i, end=' ', file=out)
                    f.tofile(out, sep=' ')
                    print(file=out)

            print('featuremap2', file=out)
            print(usage_flags2.sum(), file=out)
            for i, (f, used) in enumerate(
                    zip(feature_map2.astype('i2'), usage_flags2)):
                if used:
                    print(i, end=' ', file=out)
                    f.tofile(out, sep=' ')
                    print(file=out)

            print('feature_dwconv_weight', file=out)
            feat_dwconv_weight.astype('i2').tofile(out, sep=' ')
            print(file=out)
            print('feature_dwconv_bias', file=out)
            feat_dwconv_bias.astype('i2').tofile(out, sep=' ')
            print(file=out)

            print('policy_pwconv_layer_l1_weight', file=out)
            policy_pwconv_layer_l1_weight.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('policy_pwconv_layer_l1_bias', file=out)
            policy_pwconv_layer_l1_bias.astype('i4').tofile(out, sep=' ')
            print(file=out)
            print('policy_pwconv_layer_l2_weight', file=out)
            policy_pwconv_layer_l2_weight.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('policy_pwconv_layer_l2_bias', file=out)
            policy_pwconv_layer_l2_bias.astype('i4').tofile(out, sep=' ')
            print(file=out)

            print('value_corner_weight', file=out)
            corner_weights.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('value_corner_bias', file=out)
            corner_bias.astype('i4').tofile(out, sep=' ')
            print(file=out)
            print('value_edge_weight', file=out)
            edge_weights.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('value_edge_bias', file=out)
            edge_bias.astype('i4').tofile(out, sep=' ')
            print(file=out)
            print('value_center_weight', file=out)
            center_weights.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('value_center_bias', file=out)
            center_bias.astype('i4').tofile(out, sep=' ')
            print(file=out)
            print('value_quad_weight', file=out)
            quad_weights.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('value_quad_bias', file=out)
            quad_bias.astype('i4').tofile(out, sep=' ')
            print(file=out)

            print('value_l1_weight', file=out)
            l1_weight.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('value_l1_bias', file=out)
            l1_bias.astype('i4').tofile(out, sep=' ')
            print(file=out)
            print('value_l2_weight', file=out)
            l2_weight.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('value_l2_bias', file=out)
            l2_bias.astype('i4').tofile(out, sep=' ')
            print(file=out)
            print('value_l3_weight', file=out)
            l3_weight.astype('i1').tofile(out, sep=' ')
            print(file=out)
            print('value_l3_bias', file=out)
            l3_bias.astype('i4').tofile(out, sep=' ')
            print(file=out)

            print('policy_output_weight', file=out)
            policy_output_weight.astype('f4').tofile(out, sep=' ')
            print(file=out)
            print('policy_output_bias', file=out)
            policy_output_bias.astype('f4').tofile(out, sep=' ')
            print(file=out)
        else:
            o: io.RawIOBase = out

            # Since each quantized feature is in [-512, 511] range, they only uses 10 bits,
            # here we write compressed uint64 streams to save space.
            def write_feature_map_compressed(feature_map, feature_bits=10):
                feature_map_i16 = feature_map.astype('<i2')  # (442503, FC)
                feature_map_i16 = feature_map_i16.reshape(-1)  # (442503*FC,)
                bitmask = np.uint64(2**feature_bits - 1)
                uint64 = np.uint64(0)
                bits_used = 0
                uint64_written = 0
                for i in range(feature_map_i16.shape[0]):
                    v = feature_map_i16[i].astype(np.uint64) & bitmask
                    uint64 |= v << np.uint64(bits_used)
                    if 64 - bits_used >= feature_bits:
                        bits_used += feature_bits
                    else:
                        o.write(uint64.tobytes())
                        uint64_written += 1
                        uint64 = v >> np.uint64(64 - bits_used)
                        bits_used = feature_bits - (64 - bits_used)
                if bits_used > 0:
                    o.write(uint64.tobytes())
                    uint64_written += 1
                print(
                    f"write_feature_map_compressed: {feature_map_i16.shape[0]} -> {uint64_written} uint64"
                )

            # int16_t mapping[2][ShapeNum][FeatureDim];
            write_feature_map_compressed(feature_map1)
            write_feature_map_compressed(feature_map2)

            # int16_t feature_dwconv_weight[9][FeatureDWConvDim];
            # int16_t feature_dwconv_bias[FeatureDWConvDim];
            o.write(feat_dwconv_weight.astype('<i2').tobytes())
            o.write(feat_dwconv_bias.astype('<i2').tobytes())

            # int8_t  policy_pwconv_layer_l1_weight[(PolicyDim * 2) * FeatureDim];
            # int32_t policy_pwconv_layer_l1_bias[PolicyDim * 2];
            # int8_t  policy_pwconv_layer_l2_weight[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim) * (PolicyDim * 2)];
            # int32_t policy_pwconv_layer_l2_bias[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim)];
            o.write(policy_pwconv_layer_l1_weight.astype('<i1').tobytes())
            o.write(policy_pwconv_layer_l1_bias.astype('<i4').tobytes())
            o.write(policy_pwconv_layer_l2_weight.astype('<i1').tobytes())
            o.write(policy_pwconv_layer_l2_bias.astype('<i4').tobytes())

            # int8_t  value_corner_weight[ValueDim * FeatureDim];
            # int32_t value_corner_bias[ValueDim];
            # int8_t  value_edge_weight[ValueDim * FeatureDim];
            # int32_t value_edge_bias[ValueDim];
            # int8_t  value_center_weight[ValueDim * FeatureDim];
            # int32_t value_center_bias[ValueDim];
            # int8_t  value_quad_weight[ValueDim * ValueDim];
            # int32_t value_quad_bias[ValueDim];
            o.write(corner_weights.astype('<i1').tobytes())
            o.write(corner_bias.astype('<i4').tobytes())
            o.write(edge_weights.astype('<i1').tobytes())
            o.write(edge_bias.astype('<i4').tobytes())
            o.write(center_weights.astype('<i1').tobytes())
            o.write(center_bias.astype('<i4').tobytes())
            o.write(quad_weights.astype('<i1').tobytes())
            o.write(quad_bias.astype('<i4').tobytes())

            # int8_t  value_l1_weight[ValueDim * (FeatureDim + ValueDim * 4)];
            # int32_t value_l1_bias[ValueDim];
            # int8_t  value_l2_weight[ValueDim * ValueDim];
            # int32_t value_l2_bias[ValueDim];
            # int8_t  value_l3_weight[4 * ValueDim];
            # int32_t value_l3_bias[4];
            o.write(l1_weight.astype('<i1').tobytes())
            o.write(l1_bias.astype('<i4').tobytes())
            o.write(l2_weight.astype('<i1').tobytes())
            o.write(l2_bias.astype('<i4').tobytes())
            l3_weight = np.concatenate(
                [l3_weight,
                 np.zeros((1, l3_weight.shape[1]), dtype=np.int8)],
                axis=0)
            l3_bias = np.concatenate(
                [l3_bias, np.zeros((1, ), dtype=np.int32)], axis=0)
            o.write(l3_weight.astype('<i1').tobytes())
            o.write(l3_bias.astype('<i4').tobytes())

            # float policy_output_weight[16];
            # float policy_output_bias;
            o.write(policy_output_weight.astype('<f4').tobytes())
            o.write(policy_output_bias.astype('<f4').tobytes())
            # char  __padding_to_64bytes_1[44];
            o.write(np.zeros(44, dtype='<i1').tobytes())



def export_model(filename: str, model: Mix9Net, device, text_output=False):
    with open(filename, 'w' if text_output else 'wb') as f:
        serializer = Mix9NetSerializer(text_output)
        serializer.serialize(f, model, device)


if __name__ == '__main__':
    model = Mix9Net()
    device = torch.device('cpu')

    # TODO: load model weights

    # export as (uncompressed) binary format
    export_model('model.bin', model, device)

    # export as text format for debug purpose
    # export_model('model.txt', model, device, text_output=True)