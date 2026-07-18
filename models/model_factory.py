def build_model(name, **kwargs):

    if name == "gnn":
        from .simple_gnn import SimpleGNN
        return SimpleGNN(**kwargs)

    elif name == "gcn":
        from .gcn_model import GCNModel
        return GCNModel(**kwargs)

    elif name == "gat":
        from .temporal_gat import TemporalGAT
        return TemporalGAT(**kwargs)

    elif name == "sheaf":
        # baseline cũ
        from .sheaf_model import SheafTemporal
        return SheafTemporal(**kwargs)

    elif name in ["sheaf_conn", "sheaf_connection"]:
        # bản sheaf cải tiến mới
        from .sheaf_connection import SheafConnectionTemporal
        return SheafConnectionTemporal(**kwargs)

    else:
        raise ValueError(f"Unknown model {name}")