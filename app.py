# bh_plate_optimizer_stock_first_v9_CT_support.py
# Stock-first BH/CT plate nesting using rectpack (rotation allowed),
# staggered flange/web splices (≥300 mm offset),
# kerf/trim aware cutting,
# plate planning with tail-penalty heuristic,
# tail-merge optimization pass to collapse low-util tail plates,
# SVG + Excel outputs,
# Mill offer constraints (optional),
# KPI dashboard,
# BH input preview now also shows thickness-weight summary,
# Supports BH... and CT... profiles (CT = single flange T-profile).

import io, math, re, zipfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, DefaultDict
from collections import defaultdict

import pandas as pd
import streamlit as st

from rectpack import newPacker, PackingMode, PackingBin, MaxRectsBssf, SORT_AREA

# -------------------- Constants --------------------
SEC_PATTERN = re.compile(r"(BH|CT)(\d+)X(\d+)X(\d+)X(\d+)", re.IGNORECASE)
STEEL_DENSITY_KG_PER_M3 = 7850.0  # kg per cubic meter

# -------------------- Data Models ------------------
@dataclass
class BHRow:
    profile: str
    length_mm: int
    unit_weight_kg: float
    qty: int
    total_weight_kg: float

    # parsed geometry
    sec_type:str="BH"   # "BH" or "CT"
    H:int=0; B:int=0; tw:int=0; tf:int=0

    def parse(self):
        """
        Parse profiles like:
          BH990X640X28X45
          CT475X600X20X32
        Meaning:
          <type><H> X <B> X <tw> X <tf>
        """
        m = SEC_PATTERN.match(self.profile.strip())
        if not m:
            raise ValueError(f"Invalid profile: {self.profile}")
        self.sec_type = m.group(1).upper()
        self.H  = int(m.group(2))
        self.B  = int(m.group(3))
        self.tw = int(m.group(4))
        self.tf = int(m.group(5))

    def flange_width(self) -> int:
        # flange width is just B for both BH and CT
        return self.B

    def web_width(self) -> int:
        # For BH (I-beam type): remove top+bottom flange thickness
        # For CT (T-beam type): remove only one flange thickness
        if self.sec_type == "CT":
            return self.H - self.tf
        else:
            return self.H - 2*self.tf

@dataclass
class PlatePiece:
    kind: str            # 'flange' or 'web'
    thickness_mm: int
    width_mm: int
    length_mm: int
    qty: int
    bh_profile: str      # keep original string (BH... or CT...)

@dataclass
class SubPiece:
    parent_id:int
    index:int              # 1 or 2 when spliced
    total_len_mm:int
    length_mm:int
    width_mm:int
    thickness_mm:int
    kind:str               # 'flange'/'web'
    bh_profile:str
    splice_joint_here:bool
    joint_pos_mm: Optional[int]

@dataclass
class Placement:
    x:int; y:int; w:int; h:int
    label:str; annotate:str
    parent_id:int; sub_index:int; bh_profile:str; kind:str

@dataclass
class StockPlate:
    plate_id:str
    thickness_mm:int
    stock_width_mm:int
    stock_length_mm:int
    placements:List[Placement]=field(default_factory=list)
    trim_mm:int=0; kerf_mm:int=0
    source:str="standard"  # "inventory" or "standard"
    def utilization(self)->float:
        usable_w = self.stock_width_mm - 2*self.trim_mm
        usable_l = self.stock_length_mm - 2*self.trim_mm
        if usable_w<=0 or usable_l<=0: return 0.0
        used = 0
        x_max = self.trim_mm + usable_l
        y_max = self.trim_mm + usable_w
        for p in self.placements:
            x1 = min(p.x + p.w, x_max)
            y1 = min(p.y + p.h, y_max)
            if x1>p.x and y1>p.y:
                used += (x1 - p.x) * (y1 - p.y)
        return used / (usable_w * usable_l)

@dataclass
class InvPlate:
    t:int; w:int; l:int; qty:int; weight:float=0.0

# -------------------- Mill Offer Handling --------------------
def mill_offers_to_dict(df: pd.DataFrame) -> Dict[int, List[Tuple[int,int]]]:
    """
    Parse mill offer DataFrame -> { thickness_mm : [(W,L), (W,L), ...] }

    Expected columns:
    - "Thickness (mm)"
    - "Width (mm)"
    - "Length (mm)"
    """
    req = ["Thickness (mm)", "Width (mm)", "Length (mm)"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Mill offer missing columns: {missing}")

    d: DefaultDict[int, List[Tuple[int,int]]] = defaultdict(list)
    for _, r in df.iterrows():
        t = int(r["Thickness (mm)"])
        W = int(r["Width (mm)"])
        L = int(r["Length (mm)"])
        d[t].append((W,L))

    # Deduplicate & sort
    for t in d:
        d[t] = sorted(list(set(d[t])))
    return dict(d)

def get_allowed_plate_sizes_for_thickness(
    t:int,
    mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]
) -> List[Tuple[int,int]]:
    """
    Plate size menu for a given thickness.
    Priority:
    1. Mill offer (if provided for this thickness)
    2. Built-in defaults
    """
    if mill_sizes and t in mill_sizes:
        return mill_sizes[t]

    if t <= 45:
        widths  = [1500, 1600, 2000, 2500]
        lengths = list(range(6000, 12000, 50))
        return [(W,L) for W in widths for L in lengths]
    else:
        widths  = list(range(1200, 2500, 50))
        lengths = list(range(6000, 12000, 50))
        return [(W,L) for W in widths for L in lengths]

def usable_cap_for_thickness(t:int, trim:int, kerf:int,
                             mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]=None)->int:
    """
    Max usable 1-piece length for this thickness, based on best available plate length.
    """
    opts = get_allowed_plate_sizes_for_thickness(t, mill_sizes)
    if not opts:
        return 0
    max_L = max(L for (_,L) in opts)
    return max_L - 2*trim - kerf

# -------------------- Splice & Stagger -------------
def plan_staggered_splits_for_bh(length_mm:int,
                                 flange_t:int, web_t:int,
                                 trim_mm:int, kerf:int,
                                 mill_sizes:Optional[Dict[int,List[Tuple[int,int]]]],
                                 min_stagger_mm:int=300)->Tuple[Optional[int], Optional[int]]:
    cap_f = usable_cap_for_thickness(flange_t, trim_mm, kerf, mill_sizes)
    cap_w = usable_cap_for_thickness(web_t,    trim_mm, kerf, mill_sizes)
    need_f, need_w = length_mm > cap_f, length_mm > cap_w
    if not need_f and not need_w: return None, None

    lower = math.floor(length_mm/3); upper = math.ceil(2*length_mm/3)

    def choose_pos(cap:int)->Optional[int]:
        for a in range(lower, upper+1):
            b = length_mm - a
            if 0<a<=cap and 0<b<=cap: return a
        a = min(cap, max(lower, length_mm - cap)); b = length_mm - a
        if lower<=a<=upper and 0<a<=cap and 0<b<=cap: return a
        return None

    pos_f = choose_pos(cap_f) if need_f else None
    pos_w = choose_pos(cap_w) if need_w else None

    if need_f and need_w and pos_f is not None and pos_w is not None:
        if abs(pos_f - pos_w) < min_stagger_mm:
            # shift web first, then flange
            for delta in range(min_stagger_mm, (upper-lower)+1):
                for cand in (pos_w - delta, pos_w + delta):
                    if lower<=cand<=upper and abs(cand - pos_f) >= min_stagger_mm:
                        a=cand; b=length_mm-a
                        if 0<a<=cap_w and 0<b<=cap_w: return pos_f, cand
            for delta in range(min_stagger_mm, (upper-lower)+1):
                for cand in (pos_f - delta, pos_f + delta):
                    if lower<=cand<=upper and abs(cand - pos_w) >= min_stagger_mm:
                        a=cand; b=length_mm-a
                        if 0<a<=cap_f and 0<b<=cap_f: return cand, pos_w
    return pos_f, pos_w

# -------------------- Input Parsing ----------------
def bh_excel_to_rows(df:pd.DataFrame)->List[BHRow]:
    req = ["PROFILE","LENGTH (mm)","UNIT WEIGHT(Kg)","QTY.","TOTAL WEIGHT(Kg)"]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"BH input missing columns: {missing}")
    rows=[]
    for _,r in df.iterrows():
        br = BHRow(
            profile=str(r["PROFILE"]).strip(),
            length_mm=int(r["LENGTH (mm)"]),
            unit_weight_kg=float(r["UNIT WEIGHT(Kg)"]),
            qty=int(r["QTY."]),
            total_weight_kg=float(r["TOTAL WEIGHT(Kg)"])
        )
        br.parse()
        rows.append(br)
    return rows

def stock_excel_to_inventory(df:pd.DataFrame)->List[InvPlate]:
    req = ["T (mm)","W (mm)","L (mm)","Qty"]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Stock input missing columns: {missing}")
    inv=[]
    for _,r in df.iterrows():
        if int(r["Qty"])<=0: continue
        inv.append(InvPlate(
            int(r["T (mm)"]),
            int(r["W (mm)"]),
            int(r["L (mm)"]),
            int(r["Qty"]),
            float(r["Weight (Kg)"]) if "Weight (Kg)" in df.columns and pd.notna(r["Weight (Kg)"]) else 0.0
        ))
    return inv

def rows_to_unit_pieces(rows:List[BHRow])->List[PlatePiece]:
    """
    Convert BHRow objects into individual flange/web pieces
    BEFORE splice/stagger logic.
    Rules:
      BH section -> 2 flange + 1 web per BH
      CT section -> 1 flange + 1 web per CT (single flange T-section)
    """
    units=[]
    for br in rows:
        if br.sec_type == "CT":
            # CT -> single flange
            for _ in range(br.qty):
                units.append(PlatePiece(
                    "flange", br.tf, br.flange_width(), br.length_mm, 1, br.profile
                ))
            # CT -> web (stem)
            for _ in range(br.qty):
                units.append(PlatePiece(
                    "web", br.tw, br.web_width(), br.length_mm, 1, br.profile
                ))
        else:
            # default BH
            for _ in range(br.qty*2):
                units.append(PlatePiece(
                    "flange", br.tf, br.flange_width(), br.length_mm, 1, br.profile
                ))
            for _ in range(br.qty):
                units.append(PlatePiece(
                    "web", br.tw, br.web_width(), br.length_mm, 1, br.profile
                ))
    return units

def build_all_subpieces_with_stagger(unit_pieces:List[PlatePiece],
                                     trim_mm:int, kerf:int,
                                     mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]
                                     )->List[SubPiece]:
    by_key: DefaultDict[Tuple[str,int,str], List[PlatePiece]] = defaultdict(list)
    for p in unit_pieces:
        by_key[(p.bh_profile, p.length_mm, p.kind)].append(p)

    joint_pos: Dict[Tuple[str,int,str], List[Optional[int]]] = {}
    bh_lengths = sorted({(p.bh_profile, p.length_mm) for p in unit_pieces})

    for bh, L in bh_lengths:
        fl = by_key.get((bh,L,"flange"), [])
        wb = by_key.get((bh,L,"web"), [])
        if fl and wb:
            tf = fl[0].thickness_mm; tw = wb[0].thickness_mm
            pos_f, pos_w = plan_staggered_splits_for_bh(L, tf, tw, trim_mm, kerf, mill_sizes, 300)
        else:
            any_list = fl or wb
            tt = any_list[0].thickness_mm if any_list else 0
            cap = usable_cap_for_thickness(tt, trim_mm, kerf, mill_sizes)
            if L>cap:
                lower=math.floor(L/3); upper=math.ceil(2*L/3)
                pos = min(cap, max(lower, L-cap))
            else:
                pos = None
            pos_f = pos if fl else None
            pos_w = pos if wb else None
        joint_pos[(bh,L,"flange")] = [pos_f]*len(fl)
        joint_pos[(bh,L,"web")]    = [pos_w]*len(wb)

    subs=[]
    for (bh, L, kind), items in by_key.items():
        for i,p in enumerate(items):
            pos = joint_pos[(bh,L,kind)][i]
            cap = usable_cap_for_thickness(p.thickness_mm, trim_mm, kerf, mill_sizes)
            pid = id(p)
            if pos is None or L<=cap:
                subs.append(SubPiece(pid, 1, L, L, p.width_mm, p.thickness_mm, kind, bh, False, None))
            else:
                a=pos; b=L-a
                subs.append(SubPiece(pid, 1, L, a, p.width_mm, p.thickness_mm, kind, bh, False, pos))
                subs.append(SubPiece(pid, 2, L, b, p.width_mm, p.thickness_mm, kind, bh, True, pos))
    return subs

# -------------------- rectpack helpers -------------
def _rectpack_place(
    rects: List[SubPiece],
    stock_w: int,
    stock_l: int,
    kerf: int,
    trim: int
) -> Tuple[List[Tuple[int,int,int,int,bool,int]], Set[int]]:
    if not rects:
        return [], set()

    usable_h = stock_w - 2*trim  # Y direction
    usable_w = stock_l - 2*trim  # X direction
    if usable_h <= 0 or usable_w <= 0:
        return [], set()

    indexed_rects = list(enumerate(rects))
    indexed_rects.sort(
        key=lambda kv: kv[1].length_mm * kv[1].width_mm,
        reverse=True
    )

    packer = newPacker(
        PackingMode.Offline,
        PackingBin.BBF,
        MaxRectsBssf,
        SORT_AREA,
        True
    )

    packer.add_bin(usable_w, usable_h, count=1)

    for rid, sp in indexed_rects:
        need_x = sp.length_mm + kerf
        need_y = sp.width_mm  + kerf
        packer.add_rect(need_x, need_y, rid=rid)

    packer.pack()

    used_idx: Set[int] = set()
    placements_raw: List[Tuple[int,int,int,int,bool,int]] = []

    for (bin_idx, x, y, w, h, rid) in packer.rect_list():
        if bin_idx != 0:
            continue
        s = rects[rid]

        not_rot = (
            w == s.length_mm + kerf and
            h == s.width_mm  + kerf
        )
        rot_90 = (
            w == s.width_mm  + kerf and
            h == s.length_mm + kerf
        )

        if not_rot:
            final_w = s.length_mm
            final_h = s.width_mm
        elif rot_90:
            final_w = s.width_mm
            final_h = s.length_mm
        else:
            diff_notrot = abs(w - (s.length_mm+kerf)) + abs(h - (s.width_mm+kerf))
            diff_rot    = abs(w - (s.width_mm+kerf))   + abs(h - (s.length_mm+kerf))
            if diff_notrot <= diff_rot:
                final_w = s.length_mm
                final_h = s.width_mm
            else:
                final_w = s.width_mm
                final_h = s.length_mm

        placements_raw.append((
            int(x + trim),
            int(y + trim),
            int(final_w),
            int(final_h),
            False,  # we don't currently use rotated flag downstream
            rid
        ))
        used_idx.add(rid)

    return placements_raw, used_idx

def build_plate_from_rectpack(
    rects: List[SubPiece],
    stock_w: int,
    stock_l: int,
    kerf: int,
    trim: int,
    source: str
) -> Tuple[StockPlate, Set[int]]:

    sp = StockPlate(
        "TEMP",
        rects[0].thickness_mm if rects else 0,
        stock_w,
        stock_l,
        [],
        trim,
        kerf,
        source
    )

    placements_raw, used_idx = _rectpack_place(rects, stock_w, stock_l, kerf, trim)

    for (x, y, w, h, _rotated, rid) in placements_raw:
        s = rects[rid]
        sp.placements.append(
            Placement(
                x=x,
                y=y,
                w=w,
                h=h,
                label=f"{s.kind[:1].upper()} {s.length_mm}×{s.width_mm}×{s.thickness_mm}",
                annotate=f"{s.bh_profile}{' | WELD JOINT' if s.splice_joint_here else ''}",
                parent_id=s.parent_id,
                sub_index=s.index,
                bh_profile=s.bh_profile,
                kind=s.kind,
            )
        )
    return sp, used_idx

def _pack_once(subs:List[SubPiece],
               W:int, L:int,
               kerf:int, trim:int,
               source:str="standard") -> Tuple[Optional[StockPlate], Set[int], List[SubPiece]]:
    if not subs:
        return None, set(), []
    pl, used = build_plate_from_rectpack(subs, W, L, kerf, trim, source)
    if not used:
        return None, set(), subs[:]
    rem = [s for i,s in enumerate(subs) if i not in used]
    return pl, used, rem

def try_append_more_to_plate(
    base_plate: StockPlate,
    already_used: Set[int],
    all_rects: List[SubPiece]
) -> Tuple[StockPlate, Set[int]]:
    # placeholder for future incremental fill
    return base_plate, set()

# -------------------- Diagnostics ------------------
def explain_unplaceable(t:int, subs:List[SubPiece], trim:int, kerf:int,
                        mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]) -> List[str]:
    opts = get_allowed_plate_sizes_for_thickness(t, mill_sizes)
    if not opts:
        return [f"No standard plate sizes available for t={t}mm."]

    max_w = max(W for (W,_) in opts)
    max_l = max(L for (_,L) in opts)
    usable_w_max = max_w  - 2*trim - kerf
    usable_l_max = max_l  - 2*trim - kerf

    notes=[]
    for s in subs:
        too_wide  = s.width_mm  > usable_w_max
        too_long  = s.length_mm > usable_l_max
        if too_wide or too_long:
            msg = f"{s.kind.upper()} {s.length_mm}×{s.width_mm}×{s.thickness_mm} ({s.bh_profile})"
            reasons=[]
            if too_wide: reasons.append(f"width {s.width_mm} > max usable {usable_w_max}")
            if too_long: reasons.append(f"length {s.length_mm} > max usable {usable_l_max}")
            notes.append(msg + " — " + ", ".join(reasons))
    return notes

def _plate_waste_area(pl:StockPlate)->float:
    if pl is None:
        return 0.0
    plate_area = pl.stock_width_mm * pl.stock_length_mm
    util = pl.utilization()
    used_area = plate_area * util
    waste = plate_area - used_area
    return waste

# -------------------- Planning: Inventory First ----
def plan_on_inventory(subs_by_t:Dict[int,List[SubPiece]],
                      inventory:List[InvPlate],
                      kerf:int, trim:int,
                      min_util_pct:float,
                      priority:str,
                      start_serial:int=1) -> Tuple[List[StockPlate], Dict[int,List[SubPiece]], int, List[str]]:

    plates:List[StockPlate] = []
    remaining = {t:list(v) for t,v in subs_by_t.items()}
    serial = start_serial
    errors:List[str] = []

    inv_by_t: DefaultDict[int, List[InvPlate]] = defaultdict(list)
    for pl in inventory: inv_by_t[pl.t].append(pl)

    for t, inv_list in inv_by_t.items():
        subs = remaining.get(t, [])
        if not subs: continue

        if priority == "Largest area":
            inv_list.sort(key=lambda r: r.w*r.l, reverse=True)
        elif priority == "Closest fit":
            inv_list.sort(key=lambda r: (r.w, r.l))

        for inv in inv_list:
            count = inv.qty
            while count>0 and subs:
                plate, used = build_plate_from_rectpack(subs, inv.w, inv.l, kerf, trim, source="inventory")
                if not used:
                    break

                plate, extra_used = try_append_more_to_plate(plate, used, subs)
                used = used.union(extra_used)

                util = plate.utilization()*100
                if util < min_util_pct:
                    errors.append(f"Skipped stock {inv.w}×{inv.l} t{t} (util {util:.1f}% < {min_util_pct}%)")
                    break

                subs = [s for i,s in enumerate(subs) if i not in used]
                plate.plate_id = f"S{serial:04d}"; serial += 1
                plates.append(plate)
                count -= 1

            remaining[t] = subs
    return plates, remaining, serial, errors

# -------------------- Planning: Standard Fallback (tail-penalty heuristic) --
def plan_on_standard(remaining_by_t:Dict[int,List[SubPiece]],
                     kerf:int, trim:int,
                     start_serial:int,
                     mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]
                     ) -> Tuple[List[StockPlate], int, List[str], Dict[int,List[StockPlate]]]:

    plates_all:List[StockPlate] = []
    plates_by_t: Dict[int,List[StockPlate]] = defaultdict(list)
    serial = start_serial
    notes: List[str] = []

    TARGET_UTIL_FRACTION = 0.70

    for t, subs in sorted(remaining_by_t.items()):
        if not subs:
            continue

        size_options = get_allowed_plate_sizes_for_thickness(t, mill_sizes)

        while subs:
            best_choice = None

            for (W, L) in size_options:
                plA, usedA, remA = _pack_once(subs, W, L, kerf, trim, source="standard")
                if not usedA:
                    continue

                wasteA = _plate_waste_area(plA)

                remaining_area = 0
                for piece in remA:
                    remaining_area += piece.length_mm * piece.width_mm

                plate_area = W * L
                effective_capacity = plate_area * TARGET_UTIL_FRACTION
                if effective_capacity > 0:
                    approx_more_plates = remaining_area / effective_capacity
                else:
                    approx_more_plates = 999999

                tail_penalty_area = approx_more_plates * plate_area * (1.0 - TARGET_UTIL_FRACTION)
                score = wasteA + tail_penalty_area

                cand = {
                    "score": score,
                    "plA": plA,
                    "usedA": usedA,
                    "W": W,
                    "L": L,
                    "plate_area": plate_area,
                    "remA": remA,
                }

                if (best_choice is None
                    or cand["score"] < best_choice["score"]
                    or (cand["score"] == best_choice["score"]
                        and cand["plate_area"] < best_choice["plate_area"])):
                    best_choice = cand

            if best_choice is None:
                msg = f"No further packing possible for t={t}mm; remaining pieces: {len(subs)}"
                details = explain_unplaceable(t, subs, trim, kerf, mill_sizes)
                if details:
                    msg += "\n" + "\n".join(details)
                notes.append(msg)
                break

            pl_final = best_choice["plA"]
            used_final = best_choice["usedA"]
            subs = [s for i,s in enumerate(subs) if i not in used_final]

            pl_final.plate_id = f"N{serial:04d}"; serial += 1
            plates_all.append(pl_final)
            plates_by_t[t].append(pl_final)

        remaining_by_t[t] = subs

    return plates_all, serial, notes, plates_by_t

# -------------------- Tail Merge Optimization ------------------
def optimize_tail_plates(plates_by_t:Dict[int,List[StockPlate]],
                         kerf:int, trim:int,
                         mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]
                         ) -> Tuple[List[StockPlate], List[str]]:
    messages: List[str] = []
    final_list: List[StockPlate] = []

    def placements_to_subpieces(pl:StockPlate) -> List[SubPiece]:
        subs_local: List[SubPiece] = []
        for plc in pl.placements:
            subs_local.append(SubPiece(
                parent_id = plc.parent_id,
                index = plc.sub_index,
                total_len_mm = plc.w if plc.w >= plc.h else plc.h,
                length_mm = plc.w,
                width_mm = plc.h,
                thickness_mm = pl.thickness_mm,
                kind = plc.kind,
                bh_profile = plc.bh_profile,
                splice_joint_here = ("WELD JOINT" in plc.annotate),
                joint_pos_mm = None
            ))
        return subs_local

    for t, plist in plates_by_t.items():
        if not plist:
            continue

        std_plates = [p for p in plist if p.source=="standard"]
        inv_plates = [p for p in plist if p.source!="standard"]

        if len(std_plates) < 2:
            final_list.extend(inv_plates + std_plates)
            continue

        std_sorted = sorted(std_plates, key=lambda p: p.utilization())
        tail_group = std_sorted[:2]

        if all(p.utilization() >= 0.30 for p in tail_group):
            final_list.extend(inv_plates + std_plates)
            continue

        tail_subpieces: List[SubPiece] = []
        for p in tail_group:
            tail_subpieces.extend(placements_to_subpieces(p))

        size_options = get_allowed_plate_sizes_for_thickness(t, mill_sizes)

        best_merge = None
        for (W, L) in size_options:
            merged_plate, used_idx = build_plate_from_rectpack(
                tail_subpieces, W, L, kerf, trim, source="standard"
            )

            if len(used_idx) != len(tail_subpieces):
                continue

            util_new = merged_plate.utilization()
            old_waste = sum(_plate_waste_area(p) for p in tail_group)
            new_waste = _plate_waste_area(merged_plate)

            if (best_merge is None) or (new_waste < best_merge["new_waste"]):
                best_merge = {
                    "merged_plate": merged_plate,
                    "new_waste": new_waste,
                    "old_waste": old_waste,
                    "W": W,
                    "L": L,
                    "util_new": util_new
                }

        if best_merge is None:
            final_list.extend(inv_plates + std_plates)
            continue

        if best_merge["new_waste"] < best_merge["old_waste"]:
            kept_std = [p for p in std_plates if p not in tail_group]
            merged_plate = best_merge["merged_plate"]
            merged_plate.plate_id = "TO_RENUMBER"
            final_list.extend(inv_plates + kept_std + [merged_plate])

            messages.append(
                f"Thickness {t}mm: merged {len(tail_group)} low-util plates "
                f"into {best_merge['W']}×{best_merge['L']} (util {best_merge['util_new']*100:.1f}%)."
            )
        else:
            final_list.extend(inv_plates + std_plates)

    return final_list, messages

# -------------------- Helper: BH thickness-weight summary ----------
def summarize_weight_by_thickness(unit_pieces: List[PlatePiece]) -> pd.DataFrame:
    """
    Approx weight by thickness for all required raw pieces (before splicing).
    """
    rows_weight=[]
    for p in unit_pieces:
        thk_m   = p.thickness_mm / 1000.0
        w_m     = p.width_mm     / 1000.0
        L_m     = p.length_mm    / 1000.0
        vol_m3  = thk_m * w_m * L_m
        wt_kg   = vol_m3 * STEEL_DENSITY_KG_PER_M3
        rows_weight.append({
            "Thickness (mm)": p.thickness_mm,
            "Approx Weight (kg)": wt_kg
        })

    if not rows_weight:
        return pd.DataFrame(columns=["Thickness (mm)","Total Weight (kg)","% Share"])

    df_tmp = pd.DataFrame(rows_weight)
    grouped = df_tmp.groupby("Thickness (mm)", as_index=False)["Approx Weight (kg)"].sum()
    grouped = grouped.rename(columns={"Approx Weight (kg)":"Total Weight (kg)"})
    total_all = grouped["Total Weight (kg)"].sum()
    grouped["% Share"] = grouped["Total Weight (kg)"] / (total_all if total_all>0 else 1) * 100.0

    total_row = {
        "Thickness (mm)": "TOTAL",
        "Total Weight (kg)": total_all,
        "% Share": 100.0 if total_all>0 else 0.0
    }
    grouped = pd.concat([grouped, pd.DataFrame([total_row])], ignore_index=True)
    return grouped

# -------------------- Master Plan ------------------
def master_plan(rows:List[BHRow], inventory:List[InvPlate],
                kerf:int, trim:int,
                min_util_pct:float, priority:str,
                mill_sizes:Optional[Dict[int, List[Tuple[int,int]]]]):

    # explode BH/CT rows into individual pieces
    units = rows_to_unit_pieces(rows)

    # build subpieces after splice/stagger logic
    subs_all = build_all_subpieces_with_stagger(units, trim, kerf, mill_sizes)

    # organize by thickness
    by_t: DefaultDict[int, List[SubPiece]] = defaultdict(list)
    for s in subs_all: by_t[s.thickness_mm].append(s)

    # 1. inventory first
    plates_inv, remaining, serial, inv_errors = plan_on_inventory(
        by_t, inventory, kerf, trim, min_util_pct, priority, 1
    )

    # 2. standard plates
    plates_std, serial, std_notes, plates_by_t = plan_on_standard(
        remaining, kerf, trim, serial, mill_sizes
    )

    # 3. tail merge optimizer
    merged_list, tail_msgs = optimize_tail_plates(plates_by_t, kerf, trim, mill_sizes)

    # rebuild final list with numbering
    inventory_final = [p for p in plates_inv]
    standard_final  = [p for p in merged_list if p.source=="standard"]
    inventory_from_merge = [p for p in merged_list if p.source!="standard"]
    for p in inventory_from_merge:
        if p not in inventory_final:
            inventory_final.append(p)

    new_standard_final = []
    for p in standard_final:
        p.plate_id = f"N{serial:04d}"
        serial += 1
        new_standard_final.append(p)

    all_plates = inventory_final + new_standard_final

    # Plate Orders sheet
    orders=[]
    total_plate_weight_kg = 0.0
    for sp in all_plates:
        area_m2 = (sp.stock_width_mm*sp.stock_length_mm)*1e-6
        weight = area_m2*(sp.thickness_mm/1000.0)*STEEL_DENSITY_KG_PER_M3
        total_plate_weight_kg += weight
        orders.append({
            "Plate ID": sp.plate_id, "Source": sp.source.title(),
            "Thickness (mm)": sp.thickness_mm,
            "Width (mm)": sp.stock_width_mm, "Length (mm)": sp.stock_length_mm,
            "Est. Weight (kg)": round(weight,1),
            "Utilization %": round(sp.utilization()*100,1),
            "Weld Cuts?": any("WELD JOINT" in plc.annotate for plc in sp.placements),
        })
    order_df = pd.DataFrame(orders)

    # Procurement Summary
    procurement=[]
    if all_plates:
        df_temp = pd.DataFrame([{
            "Source":p.source.title(),
            "t":p.thickness_mm,
            "W":p.stock_width_mm,
            "L":p.stock_length_mm
        } for p in all_plates])
        for (src,t,W,L), grp in df_temp.groupby(["Source","t","W","L"]):
            tot_area = len(grp)*(W*L)*1e-6
            tot_wt   = tot_area*(t/1000.0)*STEEL_DENSITY_KG_PER_M3
            avg_u = pd.Series([pl.utilization()*100 for pl in all_plates
                               if pl.source.title()==src and pl.thickness_mm==t
                               and pl.stock_width_mm==W and pl.stock_length_mm==L]).mean()
            procurement.append({
                "Source":src,
                "Thickness (mm)":t,
                "Chosen Width (mm)":W,
                "Chosen Length (mm)":L,
                "No. of Plates":len(grp),
                "Total Area (m2)":round(tot_area,3),
                "Total Weight (kg)":round(tot_wt,1),
                "Avg Utilization %":round(avg_u,1)
            })
    procurement_df = pd.DataFrame(procurement)

    # BH Pieces derived table
    bh_pieces_df = pd.DataFrame([{
        "BH/CT": p.bh_profile,
        "Kind": p.kind,
        "t (mm)": p.thickness_mm,
        "w (mm)": p.width_mm,
        "L (mm)": p.length_mm,
        "Qty": 1
    } for p in units])

    # Shop Marking table
    shop_df = build_shop_marking_df(all_plates)

    # messages
    messages = []
    messages.extend(inv_errors)
    messages.extend(std_notes)
    messages.extend(tail_msgs)

    # KPIs:
    total_bh_weight_kg = sum(r.total_weight_kg for r in rows)
    utilization_pct_overall = (total_bh_weight_kg / total_plate_weight_kg * 100.0) if total_plate_weight_kg>0 else 0.0

    thickness_summary_df = summarize_weight_by_thickness(units)

    return (
        all_plates,
        order_df,
        procurement_df,
        bh_pieces_df,
        shop_df,
        subs_all,
        messages,
        total_bh_weight_kg,
        total_plate_weight_kg,
        utilization_pct_overall,
        thickness_summary_df
    )

# -------------------- Shop Export (Excel) ----------
def build_shop_marking_df(plates:List[StockPlate])->pd.DataFrame:
    rows=[]
    for sp in plates:
        usable_w = sp.stock_width_mm - 2*sp.trim_mm
        usable_l = sp.stock_length_mm - 2*sp.trim_mm
        shelf_map = {y:i+1 for i,y in enumerate(sorted({p.y for p in sp.placements}))}
        for p in sp.placements:
            m = re.match(r"([FW])\s+(\d+)×(\d+)×(\d+)", p.label)
            kind = 'flange' if (m and m.group(1).upper()=='F') else ('web' if m else '')
            orig_L = int(m.group(2)) if m else None
            orig_W = int(m.group(3)) if m else None
            piece_t = int(m.group(4)) if m else sp.thickness_mm

            rotated = ""
            if m:
                if orig_L == p.w and orig_W == p.h:
                    rotated = "No"
                elif orig_L == p.h and orig_W == p.w:
                    rotated = "Yes"
                else:
                    rotated = "Check"

            bh = p.annotate.split(' | ')[0]

            rows.append({
                "Plate ID": sp.plate_id,
                "Source": sp.source.title(),
                "Plate t (mm)": sp.thickness_mm,
                "Plate Size (W×L)": f"{sp.stock_width_mm}×{sp.stock_length_mm}",
                "Usable (W×L)": f"{usable_w}×{usable_l}",
                "BH/CT": bh,
                "Kind": kind,
                "Piece t (mm)": piece_t,
                "Orig L (mm)": orig_L,
                "Orig W (mm)": orig_W,
                "Placed L (mm)": p.w,
                "Placed W (mm)": p.h,
                "Rotated 90°": rotated,
                "Weld Joint?": "Yes" if "WELD JOINT" in p.annotate else "No",
                "X0 (mm)": p.x,
                "Y0 (mm)": p.y,
                "X1 (mm)": p.x+p.w,
                "Y1 (mm)": p.y+p.h,
                "Shelf #": shelf_map.get(p.y,""),
                "Parent ID": p.parent_id,
                "Sub Index": p.sub_index,
            })
    return pd.DataFrame(rows)

def order_sheet_to_excel(order_df:pd.DataFrame,
                         procurement_df:pd.DataFrame,
                         shop_df:pd.DataFrame,
                         bh_pieces_df:pd.DataFrame)->bytes:
    mem = io.BytesIO()
    with pd.ExcelWriter(mem, engine="xlsxwriter") as w:
        order_df.to_excel(w, index=False, sheet_name="Plate Orders")
        procurement_df.to_excel(w, index=False, sheet_name="Procurement Summary")
        shop_df.to_excel(w, index=False, sheet_name="Shop Marking (Detailed)")
        bh_pieces_df.to_excel(w, index=False, sheet_name="BH Pieces (Derived)")
        for s in ("Plate Orders","Procurement Summary","Shop Marking (Detailed)","BH Pieces (Derived)"):
            ws = w.sheets[s]
            ws.set_column(0, 2, 18); ws.set_column(3, 10, 20); ws.set_column(11, 20, 22)
    mem.seek(0); return mem.read()

# -------------------- SVG Drawing ------------------
def svg_marking_2d(plate:StockPlate, kerf:int, trim:int)->str:
    SCALE = 0.14
    m = 40
    W = int(plate.stock_length_mm*SCALE) + 2*m
    H = int(plate.stock_width_mm*SCALE) + 2*m
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    parts.append(
        f'<rect x="{m}" y="{m}" '
        f'width="{int(plate.stock_length_mm*SCALE)}" '
        f'height="{int(plate.stock_width_mm*SCALE)}" '
        f'fill="#f6f6f6" stroke="#111" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{m}" y="{m-12}" font-family="monospace" font-size="16">'
        f'{plate.plate_id} [{plate.source}] '
        f't{plate.thickness_mm} '
        f'{plate.stock_width_mm}×{plate.stock_length_mm} '
        f'Util {plate.utilization()*100:.1f}%'
        f'</text>'
    )
    tx=m+int(trim*SCALE); ty=m+int(trim*SCALE)
    tw=int((plate.stock_length_mm-2*trim)*SCALE); th=int((plate.stock_width_mm-2*trim)*SCALE)
    parts.append(
        f'<rect x="{tx}" y="{ty}" width="{tw}" height="{th}" '
        f'fill="none" stroke="#888" stroke-dasharray="6,6" stroke-width="1.5"/>'
    )
    for plc in plate.placements:
        x=m+int(plc.x*SCALE); y=m+int(plc.y*SCALE); w=int(plc.w*SCALE); h=int(plc.h*SCALE)
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="#dff1ff" stroke="#0b79d0" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x+6}" y="{y+18}" font-size="12" font-family="monospace">{plc.label}</text>'
        )
        parts.append(
            f'<text x="{x+6}" y="{y+34}" font-size="12" font-family="monospace">{plc.annotate}</text>'
        )
    parts.append('</svg>')
    return "\n".join(parts)

def svg_splice_view(bh_profile:str, kind:str, total_len:int,
                    seg1_len:int, seg2_len:int, joint_pos:int,
                    plate1_id:str, plate2_id:str)->str:
    SCALE = 0.14
    m=30; title_h=24; bar_h_mm=250
    W = int(total_len*SCALE) + 2*m
    H = int(bar_h_mm*SCALE) + title_h + 2*m
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    parts.append(
        f'<text x="{m}" y="{m+title_h-6}" font-family="monospace" font-size="16">'
        f'{bh_profile} — {kind.upper()} — Total {total_len} mm'
        f'</text>'
    )
    x0=m; y0=m+title_h; bar_h=int(bar_h_mm*SCALE)
    w1=int(seg1_len*SCALE)
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{w1}" height="{bar_h}" '
        f'fill="#dff1ff" stroke="#0b79d0"/>'
    )
    parts.append(
        f'<text x="{x0+6}" y="{y0+18}" font-size="12" font-family="monospace">'
        f'Seg1 {seg1_len} mm | Plate {plate1_id}'
        f'</text>'
    )
    jx = x0 + int(joint_pos*SCALE)
    parts.append(
        f'<line x1="{jx}" y1="{y0}" x2="{jx}" y2="{y0+bar_h}" '
        f'stroke="#cc0000" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{jx+6}" y="{y0+bar_h//2}" font-size="12" font-family="monospace">'
        f'WELD @ {joint_pos} mm'
        f'</text>'
    )
    w2=int(seg2_len*SCALE)
    parts.append(
        f'<rect x="{x0+w1}" y="{y0}" width="{w2}" height="{bar_h}" '
        f'fill="#e8ffe1" stroke="#2b8a3e"/>'
    )
    parts.append(
        f'<text x="{x0+w1+6}" y="{y0+18}" font-size="12" font-family="monospace">'
        f'Seg2 {seg2_len} mm | Plate {plate2_id}'
        f'</text>'
    )
    parts.append('</svg>')
    return "\n".join(parts)

def collect_splice_views(plates:List[StockPlate], subs_all:List[SubPiece])->List[Tuple[str,str,str]]:
    where: Dict[Tuple[int,int], str] = {}
    for sp in plates:
        for plc in sp.placements:
            where[(plc.parent_id, plc.sub_index)] = sp.plate_id
    by_parent: DefaultDict[int, List[SubPiece]] = defaultdict(list)
    for s in subs_all: by_parent[s.parent_id].append(s)

    views=[]
    for pid, chunks in by_parent.items():
        if len(chunks)!=2:
            continue
        a=[c for c in chunks if c.index==1][0]
        b=[c for c in chunks if c.index==2][0]
        if a.joint_pos_mm is None:
            continue
        p1 = where.get((a.parent_id,1),"—")
        p2 = where.get((a.parent_id,2),"—")
        svg = svg_splice_view(a.bh_profile, a.kind, a.total_len_mm,
                              a.length_mm, b.length_mm, a.joint_pos_mm,
                              p1, p2)
        title = f"{a.bh_profile} — {a.kind.upper()} splice ({a.total_len_mm} → {a.length_mm}+{b.length_mm})"
        views.append((title, svg, f"{pid}-{a.kind}"))
    return views

# -------------------- Marking ZIP (t_w_l filenames) ---------------
def make_zip_of_svgs(plates:List[StockPlate], kerf:int, trim:int)->bytes:
    mem=io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for sp in plates:
            z.writestr(
                f"markings/{sp.plate_id}_{sp.source}_t{sp.thickness_mm}_{sp.stock_width_mm}_{sp.stock_length_mm}.svg",
                svg_marking_2d(sp, kerf, trim)
            )
        z.writestr(
            "markings/README.txt",
            "SVGs with usable-window dashed; labels show L×W×t & BH/CT. "
            "Title shows source [inventory|standard]."
        )
    mem.seek(0); return mem.read()

# -------------------- Streamlit App --------------
def app():
    st.set_page_config(
        page_title="BH / CT Plate Nesting & Procurement Planner",
        layout="wide"
    )
    st.title("BH / CT Plate Nesting & Procurement Planner")
    st.caption(
        "• Stock-first nesting • Splice staggering • Weld joint map • Procurement excel • "
        "Tail merge optimizer • Mill offer control • CT single-flange support"
    )

    # ---------- Sidebar ----------
    with st.sidebar:
        st.subheader("Settings")
        kerf = st.number_input("Cutting kerf (mm)", 0, 10, 4)
        trim = st.number_input("Trim on each edge (mm)", 0, 200, 5)
        min_util = st.slider("Min Utilization % for Stock", 0, 100, 55)
        priority = st.selectbox("Stock priority", ["Largest area", "Closest fit"])
        st.caption("Flange/Web splices staggered ≥ 300 mm when both require splicing.")
        st.markdown("---")
        f_bh = st.file_uploader("Upload BH/CT Excel", type=["xlsx","xls","csv"], key="bh")
        f_stock = st.file_uploader("Upload Stock Plates (optional)", type=["xlsx","xls","csv"], key="stock")
        f_mill = st.file_uploader("Mill Plate Offer (optional)", type=["xlsx","xls","csv"], key="mill")
        st.caption("Mill Offer columns: Thickness (mm), Width (mm), Length (mm)")

    if not f_bh:
        st.info("Upload BH/CT Excel to begin.")
        st.stop()

    # ---------- Read inputs ----------
    df_bh = pd.read_csv(f_bh) if f_bh.name.lower().endswith(".csv") else pd.read_excel(f_bh)

    df_stock = None
    if f_stock is not None:
        df_stock = pd.read_csv(f_stock) if f_stock.name.lower().endswith(".csv") else pd.read_excel(f_stock)

    df_mill = None
    mill_sizes: Optional[Dict[int, List[Tuple[int,int]]]] = None
    if f_mill is not None:
        df_mill = pd.read_csv(f_mill) if f_mill.name.lower().endswith(".csv") else pd.read_excel(f_mill)
        try:
            mill_sizes = mill_offers_to_dict(df_mill)
        except Exception as e:
            st.error(f"Mill offer error: {e}")
            st.stop()

    # ---------- Parse BH/CT ----------
    try:
        rows = bh_excel_to_rows(df_bh)  # rows contain sec_type now (BH or CT)
    except Exception as e:
        st.error(f"BH/CT input error: {e}")
        st.stop()

    # ---------- Parse Stock ----------
    inventory: List[InvPlate] = []
    if df_stock is not None:
        try:
            inventory = stock_excel_to_inventory(df_stock)
        except Exception as e:
            st.error(f"Stock input error: {e}")
            st.stop()

    # ---------- Run planner ----------
    (
        plates,
        order_df,
        procurement_df,
        bh_pieces_df,
        shop_df,
        subs_all,
        messages,
        total_bh_weight_kg,
        total_plate_weight_kg,
        utilization_pct_overall,
        thickness_summary_df
    ) = master_plan(
        rows, inventory, kerf, trim, min_util, priority, mill_sizes
    )

    # ---------- BH/CT reference table ----------
    df_bh_ref = pd.DataFrame([{
        "Section": r.profile,
        "Type": r.sec_type,
        "H": r.H,
        "B": r.B,
        "tw": r.tw,
        "tf": r.tf,
        "L (mm)": r.length_mm,
        "Qty": r.qty,
        "Total Weight (kg)": r.total_weight_kg
    } for r in rows])

    # ---------- KPI Matrix ----------
    st.subheader("Overall Material KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Total BH/CT Required Weight (kg)",
        value=f"{total_bh_weight_kg:,.1f}"
    )
    col2.metric(
        label="Total Plate Weight Planned (kg)",
        value=f"{total_plate_weight_kg:,.1f}"
    )
    col3.metric(
        label="Material Utilization (%)",
        value=f"{utilization_pct_overall:,.1f}%"
    )

    st.markdown("---")

    # ---------- Input Preview Expanders ----------
    with st.expander("Input Preview — BH / CT", expanded=False):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**BH/CT Input (first 50 rows)**")
            st.dataframe(df_bh.head(50), use_container_width=True)

        with c2:
            st.markdown("**Thickness-wise Approx Weight Requirement**")
            st.dataframe(thickness_summary_df, use_container_width=True)

    if df_stock is not None:
        with st.expander("Input Preview — Stock Plates", expanded=False):
            st.dataframe(df_stock.head(50), use_container_width=True)

    if df_mill is not None:
        with st.expander("Input Preview — Mill Plate Offer", expanded=False):
            st.dataframe(df_mill.head(100), use_container_width=True)

    # ---------- Tabs ----------
    tabs = st.tabs([
        "Procurement",
        "Shop Marking",
        "Splice Joint Views",
        "Marking Drawings",
        "Exceptions / Notes"
    ])

    with tabs[0]:
        st.markdown("### Plate Orders")
        st.dataframe(order_df, use_container_width=True)

        st.markdown("### Procurement Summary")
        st.dataframe(procurement_df, use_container_width=True)

        if mill_sizes:
            allowed_view = []
            for t, pairs in sorted(mill_sizes.items()):
                for (W,L) in pairs:
                    allowed_view.append({
                        "Thickness (mm)": t,
                        "Allowed Width (mm)": W,
                        "Allowed Length (mm)": L
                    })
            df_allowed = pd.DataFrame(allowed_view)
            with st.expander("Mill Offer Used (Allowed Plate Sizes)", expanded=False):
                st.dataframe(df_allowed, use_container_width=True)
        else:
            st.caption("No Mill Offer uploaded. Using default standard plate assumptions.")

        xls = order_sheet_to_excel(order_df, procurement_df, shop_df, bh_pieces_df)
        st.download_button(
            "📥 Download Excel (Orders + Shop + BH/CT)",
            data=xls,
            file_name="BH_CT_plate_orders_and_shop_stock_first.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with tabs[1]:
        st.markdown("### Shop Marking Detailed Table")
        st.dataframe(shop_df, use_container_width=True)

    with tabs[2]:
        st.markdown("### Separate Drawings for Splice Joints (Staggered)")
        views = collect_splice_views(plates, subs_all)
        if not views:
            st.info("No spliced members found.")
        else:
            for title, svg, key in views:
                st.markdown(f"**{title}**")
                st.components.v1.html(svg, height=200, scrolling=False)
                st.markdown("---")

    with tabs[3]:
        st.markdown("### Marking Drawings (All Plates)")
        z = make_zip_of_svgs(plates, kerf, trim)
        st.download_button(
            "📦 Download All Markings (ZIP)",
            data=z,
            file_name="markings_stock_first.zip",
            mime="application/zip"
        )
        st.caption("Preview (first few):")
        for sp in plates[:min(300, len(plates))]:
            st.markdown(
                f"**{sp.plate_id}** [{sp.source}] — "
                f"t{sp.thickness_mm} | "
                f"{sp.stock_width_mm}×{sp.stock_length_mm} | "
                f"Util {sp.utilization()*100:.1f}%"
            )
            svg = svg_marking_2d(sp, kerf, trim)
            st.components.v1.html(svg, height=520, scrolling=True)

    with tabs[4]:
        if messages:
            st.warning("\n".join(messages))
        else:
            st.success("No exceptions.")

if __name__ == "__main__":
    app()
