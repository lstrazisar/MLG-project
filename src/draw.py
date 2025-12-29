from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def draw_bond_importance(smiles, bond_imp, size=(450, 350)):
    """
    Make a drawing of the molecule with bonds colored by importance.

    :param smiles: SMILES string of the molecule
    :param bond_imp: list of ((atom_idx1, atom_idx2), importance_value
    :param size: tuple of (width, height) for the image size
    :return: PNG image data as bytes
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # normalize
    vals = [float(delta[0]) for _, delta in bond_imp]
    vmax = max(abs(v) for v in vals) + 1e-12

    bond_colors = {}
    highlight_bonds = []

    for (a, b), delta in bond_imp:
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            continue
        bid = bond.GetIdx()
        v = float(delta[0]) / vmax
        # red = important
        bond_colors[bid] = (1.0, 1.0 - abs(v), 1.0 - abs(v))
        highlight_bonds.append(bid)

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def collapse_H_to_heavy(smiles: str, atom_imp_withHs):
    """
    Given a SMILES and atom importance values including Hs, return a mol without Hs and
    importance values for heavy atoms where each heavy atom's importance includes the importance
    of its attached Hs.

    :param smiles: SMILES string of the molecule
    :param atom_imp_withHs: list of importance values for each atom including Hs
    :return: (mol without Hs, list of importance values for heavy atoms)
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    imp = [float(x) for x in atom_imp_withHs]

    heavy_ids = []
    heavy_imp = []
    idx_map = {}  # old idx -> heavy idx

    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 1:
            idx_map[a.GetIdx()] = len(heavy_ids)
            heavy_ids.append(a.GetIdx())
            heavy_imp.append(imp[a.GetIdx()])

    # add each H importance to its neighbor heavy atom
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            nbrs = list(a.GetNeighbors())
            if len(nbrs) == 1:
                heavy = nbrs[0].GetIdx()
                if heavy in idx_map:
                    heavy_imp[idx_map[heavy]] += imp[a.GetIdx()]

    mol_noHs = Chem.MolFromSmiles(smiles)  # no explicit Hs
    return mol_noHs, heavy_imp


def _normalize(vals, eps=1e-12):
    """
    Normalize vals to [-1, 1] by dividing by max abs value.

    :param vals: list of float values
    :param eps: small value to avoid division by zero
    :return: (normalized values, vmax)
    """
    vmax = float(max(abs(v) for v in vals)) if len(vals) else 1.0
    vmax = max(vmax, eps)
    return [v / vmax for v in vals], vmax


def _red_blue_color(v):
    """
    Map a value in [-1, 1] to a color from blue (negative) to white (zero) to red (positive).

    :param v: float value in [-1, 1]
    :return: (r, g, b) tuple with values in [0, 1]
    """
    v = max(-1.0, min(1.0, float(v)))
    if v >= 0:
        # white -> red
        return (1.0, 1.0 - v, 1.0 - v)
    else:
        # white -> blue
        vv = -v
        return (1.0 - vv, 1.0 - vv, 1.0)


def draw_mol_importance(smiles, atom_importance, title="", size=(450, 350), legend=None, add_hs=True):
    """
    Make a drawing of the molecule with atoms colored by importance.

    :param smiles: SMILES string of the molecule
    :param atom_importance: list of importance values for each atom
    :param title: title string to include in the legend
    :param size: tuple of (width, height)
    :param legend: custom legend string (overrides title)
    :param add_hs: whether to add hydrogens to the molecule before visualization
    :return: PNG image data as bytes
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if add_hs:
        mol = Chem.AddHs(mol)

    imp = [float(x) for x in atom_importance]
    if len(imp) != mol.GetNumAtoms():
        raise ValueError(
            f"Importance length ({len(imp)}) != num atoms in RDKit mol with Hs ({mol.GetNumAtoms()}). "
            f"Either visualize with Hs or adjust how you build the graph."
        )

    imp_norm, vmax = _normalize(imp)
    atom_colors = {i: _red_blue_color(v) for i, v in enumerate(imp_norm)}
    highlight_atoms = list(range(mol.GetNumAtoms()))

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addAtomIndices = False  # set True if you want atom ids
    if legend is None:
        legend = f"{title} (max |imp|={vmax:.3g})" if title else f"max |imp|={vmax:.3g}"

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=legend,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


if __name__ == "__main__":
    pass
