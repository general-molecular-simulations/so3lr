from so3lr.mlff.nn.embed import AtomTypeEmbedSparse, ChargeEmbedSparse, SpinEmbedSparse


def make_embedding_modules(
        num_features: int,
        use_charge_embed: bool,
        use_spin_embed: bool
):
    embedding_modules = []
    atom_type_embed = AtomTypeEmbedSparse(
        num_features=num_features,
        prop_keys=None
    )
    embedding_modules.append(atom_type_embed)

    # Embed the total charge.
    if use_charge_embed:
        charge_embed = ChargeEmbedSparse(
            num_features=num_features,
            prop_keys=None
        )
        embedding_modules.append(charge_embed)

    # Embed the number of unpaired electrons.
    if use_spin_embed:
        spin_embed = SpinEmbedSparse(
            num_features=num_features,
            prop_keys=None
        )
        embedding_modules.append(spin_embed)

    return embedding_modules
