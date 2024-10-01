from arch.hourglass import image_transformer_v2 as itv2
from arch.hourglass.image_transformer_v2 import ImageTransformerDenoiserModelV2
from arch.swinir.swinir import SwinIR


def create_arch(arch, condition_channels=0):
    # arch should be, e.g., swinir_XL, or hdit_XL
    arch_name, arch_size = arch.split('_')
    arch_config = arch_configs[arch_name][arch_size].copy()
    arch_config['in_channels'] += condition_channels
    return arch_name_to_object[arch_name](**arch_config)


arch_configs = {
    'hdit': {
        "ImageNet256Sp4": {
            'in_channels': 3,
            'out_channels': 3,
            'widths': [256, 512, 1024],
            'depths': [2, 2, 8],
            'patch_size': [4, 4],
            'self_attns': [
                {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
                {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
                {"type": "global", "d_head": 64}
            ],
            'mapping_depth': 2,
            'mapping_width': 768,
            'dropout_rate': [0, 0, 0],
            'mapping_dropout_rate': 0.0
        },
        "XL2": {
            'in_channels': 3,
            'out_channels': 3,
            'widths': [384, 768],
            'depths': [2, 11],
            'patch_size': [4, 4],
            'self_attns': [
                {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
                {"type": "global", "d_head": 64}
            ],
            'mapping_depth': 2,
            'mapping_width': 768,
            'dropout_rate': [0, 0],
            'mapping_dropout_rate': 0.0
        }

    },
    'swinir': {
        "M": {
            'in_channels': 3,
            'out_channels': 3,
            'embed_dim': 120,
            'depths': [6, 6, 6, 6, 6],
            'num_heads': [6, 6, 6, 6, 6],
            'resi_connection': '1conv',
            'sf': 8

        },
        "L": {
            'in_channels': 3,
            'out_channels': 3,
            'embed_dim': 180,
            'depths': [6, 6, 6, 6, 6, 6, 6, 6],
            'num_heads': [6, 6, 6, 6, 6, 6, 6, 6],
            'resi_connection': '1conv',
            'sf': 8
        },
    },
}


def create_swinir_model(in_channels, out_channels, embed_dim, depths, num_heads, resi_connection,
                        sf):
    return SwinIR(
        img_size=64,
        patch_size=1,
        in_chans=in_channels,
        num_out_ch=out_channels,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=8,
        mlp_ratio=2,
        sf=sf,
        img_range=1.0,
        upsampler="nearest+conv",
        resi_connection=resi_connection,
        unshuffle=True,
        unshuffle_scale=8
    )


def create_hdit_model(widths,
                      depths,
                      self_attns,
                      dropout_rate,
                      mapping_depth,
                      mapping_width,
                      mapping_dropout_rate,
                      in_channels,
                      out_channels,
                      patch_size
                      ):
    assert len(widths) == len(depths)
    assert len(widths) == len(self_attns)
    assert len(widths) == len(dropout_rate)
    mapping_d_ff = mapping_width * 3
    d_ffs = []
    for width in widths:
        d_ffs.append(width * 3)

    levels = []
    for depth, width, d_ff, self_attn, dropout in zip(depths, widths, d_ffs, self_attns, dropout_rate):
        if self_attn['type'] == 'global':
            self_attn = itv2.GlobalAttentionSpec(self_attn.get('d_head', 64))
        elif self_attn['type'] == 'neighborhood':
            self_attn = itv2.NeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
        elif self_attn['type'] == 'shifted-window':
            self_attn = itv2.ShiftedWindowAttentionSpec(self_attn.get('d_head', 64), self_attn['window_size'])
        elif self_attn['type'] == 'none':
            self_attn = itv2.NoAttentionSpec()
        else:
            raise ValueError(f'unsupported self attention type {self_attn["type"]}')
        levels.append(itv2.LevelSpec(depth, width, d_ff, self_attn, dropout))
    mapping = itv2.MappingSpec(mapping_depth, mapping_width, mapping_d_ff, mapping_dropout_rate)
    model = ImageTransformerDenoiserModelV2(
        levels=levels,
        mapping=mapping,
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=patch_size,
        num_classes=0,
        mapping_cond_dim=0,
    )

    return model


arch_name_to_object = {
    'hdit': create_hdit_model,
    'swinir': create_swinir_model,
}
