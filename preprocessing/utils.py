import math
import re

from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock


def get_atomic_props_block(composition, oxi=False):
    noble_vdw_radii = {
        "He": 1.40,
        "Ne": 1.54,
        "Ar": 1.88,
        "Kr": 2.02,
        "Xe": 2.16,
        "Rn": 2.20,
    }

    allen_electronegativity = {
        "He": 4.16,
        "Ne": 4.79,
        "Ar": 3.24,
    }

    def _format(val):
        if val is None:
            return " 0.0000"
        try:
            f_val = float(val)
            if math.isnan(f_val):
                return " 0.0000"
            return f"{f_val: .4f}"
        except (ValueError, TypeError):
            return " 0.0000"

    def _format_X(elem):
        if math.isnan(elem.X) and str(elem) in allen_electronegativity:
            return _format(allen_electronegativity[str(elem)])
        return _format(elem.X)

    def _format_radius(elem):
        radius = elem.atomic_radius
        if radius is None and str(elem) in noble_vdw_radii:
            radius = noble_vdw_radii[str(elem)]
        return _format(radius)

    def _format_ionic_radius(elem):
        # Multi-stage fallback to prevent "AttributeError"
        radius = getattr(elem, "average_ionic_radius", None)
        if radius is None:
            radius = getattr(elem, "ionic_radius", None)
        if radius is None:
            radius = elem.atomic_radius  # Final fallback
        return _format(radius)

    props = {str(el): (_format_X(el), _format_radius(el), _format_ionic_radius(el))
             for el in sorted(composition.elements)}
    
    data = {}
    data["_atom_type_symbol"] = list(props)
    data["_atom_type_electronegativity"] = [v[0] for v in props.values()]
    data["_atom_type_radius"] = [v[1] for v in props.values()]
    data["_atom_type_ionic_radius"] = [v[2] for v in props.values()]

    loop_vals = [
        "_atom_type_symbol",
        "_atom_type_electronegativity",
        "_atom_type_radius",
        "_atom_type_ionic_radius"
    ]

    if oxi:
        symbol_to_oxinum = {}
        for el in sorted(composition.elements):
            # Safely get oxidation state or default to 0
            o_state = float(getattr(el, "oxi_state", 0))
            # Safely get ionic radius for that specific state, or fallback
            i_rad = getattr(el, "ionic_radius", None)
            if i_rad is None:
                i_rad = getattr(el, "average_ionic_radius", el.atomic_radius)
            symbol_to_oxinum[str(el)] = (o_state, _format(i_rad))

        data["_atom_type_oxidation_number"] = [v[0] for v in symbol_to_oxinum.values()]
        data["_atom_type_ionic_radius"] = [v[1] for v in symbol_to_oxinum.values()]
        loop_vals.append("_atom_type_oxidation_number")

    loops = [loop_vals]
    return str(CifBlock(data, loops, "")).replace("data_\n", "")


def extract_numeric_property(cif_str, prop, numeric_type=float):
    match = re.search(rf"{prop}\s+([-+]?[.0-9]+)", cif_str)
    if match:
        return numeric_type(match.group(1))
    # Return a default if not found to avoid crashing the whole pipeline
    if numeric_type == int: return 0
    return 0.0

def extract_volume(cif_str):
    return extract_numeric_property(cif_str, "_cell_volume")

def extract_formula_nonreduced(cif_str):
    match = re.search(r"_chemical_formula_sum\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract _chemical_formula_sum")

def extract_space_group_symbol(cif_str):
    match = re.search(r"_symmetry_space_group_name_H-M\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract space group")

def extract_formula_units(cif_str):
    return extract_numeric_property(cif_str, "_cell_formula_units_Z", numeric_type=int)

def semisymmetrize_cif(cif_str):
    return re.sub(
        r"(_symmetry_equiv_pos_as_xyz\n)(.*?)(?=\n(?:\S| \S))",
        r"\1  1  'x, y, z'",
        cif_str,
        flags=re.DOTALL
    )

def replace_data_formula_with_nonreduced_formula(cif_str):
    pattern = r"_chemical_formula_sum\s+(.+)\n"
    pattern_2 = r"(data_)(.*?)(\n)"
    match = re.search(pattern, cif_str)
    if match:
        chemical_formula = match.group(1).replace("'", "").replace(" ", "")
        return re.sub(pattern_2, r'\1' + chemical_formula + r'\3', cif_str)
    return cif_str
    
def add_atomic_props_block(cif_str, oxi=False):
    try:
        formula = extract_formula_nonreduced(cif_str)
        comp = Composition(formula)
        
        if oxi:
            try:
                comp.add_charges_from_oxi_state_guesses()
            except:
                oxi = False # Fallback to neutral properties if guessing fails

        block = get_atomic_props_block(composition=comp, oxi=oxi)
        pattern = r"_symmetry_space_group_name_H-M"
        match = re.search(pattern, cif_str)

        if match:
            start_pos = match.start()
            return cif_str[:start_pos] + block + "\n" + cif_str[start_pos:]
        return cif_str
    except Exception:
        return cif_str

def round_numbers(cif_str, decimal_places=4):
    pattern = r"([-+]?\d*\.\d+([eE][-+]?\d+)?)"
    def replace(match):
        try:
            return f"{float(match.group(1)):.{decimal_places}f}"
        except:
            return match.group(1)
    return re.sub(pattern, replace, cif_str)