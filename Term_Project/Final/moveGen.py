
from __future__ import annotations

import collections
import copy
import dataclasses
import enum
import math
import re
import itertools
import typing

from typing import Dict, Iterable, Callable, Tuple,  Iterator, List, ClassVar, Hashable, Type,  Counter, Mapping, Optional, SupportsInt, Generic, TypeVar, Union

try:
    from typing import Literal
    Passant = Literal["legal", "fen", "xfen"]
except ImportError:
    Passant = str  


Color = bool
COLORS = [WHITE, BLACK] = [True, False]
COLOR_NAMES = ["black", "white"]

PieceType = int
pieceTypes = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
pieceSymbolS = [None, "p", "n", "b", "r", "q", "k"]
pieceNames = [None, "pawn", "knight", "bishop", "rook", "queen", "king"]


def pieceSymbol(Piecetype: PieceType) -> str:
    return typing.cast(str, pieceSymbolS[Piecetype])


def piece_name(Piecetype: PieceType) -> str:
    return typing.cast(str, pieceNames[Piecetype])


asci_dict = {
    "R": "♖", "r": "♜",
    "N": "♘", "n": "♞",
    "B": "♗", "b": "♝",
    "Q": "♕", "q": "♛",
    "K": "♔", "k": "♚",
    "P": "♙", "p": "♟",
}

FileName = ["a", "b", "c", "d", "e", "f", "g", "h"]

RankName = ["1", "2", "3", "4", "5", "6", "7", "8"]

#starter
fenStart = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
fenBoard = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


@dataclasses.dataclass
class Outcome:
    termination: Termination
    winner: Optional[Color]

    def result(self) -> str:
        return "1/2-1/2" if self.winner is None else ("1-0" if self.winner else "0-1")


class InvalidMoveError(ValueError):
    """not worded properly"""
class IllegalMoveError(ValueError):
    """this move is illegal"""
class AmbiguousMoveError(ValueError):
    """this move is no able to be made by this piece"""


SquareBoard = int

#Equal 1 to 64
SquareBoardS = [
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
] = range(64)

SquareBoardName = [f + r for r in RankName for f in FileName]


def parse_SquareBoard(name: str) -> SquareBoard:
    return SquareBoardName.index(name)


def SquareBoard_name(SquareBoard: SquareBoard) -> str:
    return SquareBoardName[SquareBoard]


def SquareBoard(file_index: int, rank_index: int) -> SquareBoard:
    return rank_index * 8 + file_index


def SquareBoard_file(SquareBoard: SquareBoard) -> int:
    return SquareBoard & 7


def SquareBoard_rank(SquareBoard: SquareBoard) -> int:
    return SquareBoard >> 3


def SquareBoard_distance(a: SquareBoard, b: SquareBoard) -> int:
    return max(abs(SquareBoard_file(a) - SquareBoard_file(b)), abs(SquareBoard_rank(a) - SquareBoard_rank(b)))


def SquareBoard_mirror(SquareBoard: SquareBoard) -> SquareBoard:
    return SquareBoard ^ 0x38


SquareBoardS_180 = [SquareBoard_mirror(sq) for sq in SquareBoardS]


BitBoard = int
BB_EMPTY = 0
BB_ALL = 0xffff_ffff_ffff_ffff

BB_SquareBoardS = [
    BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1,
    BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2,
    BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3,
    BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4,
    BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5,
    BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6,
    BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7,
    BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8,
] = [1 << sq for sq in SquareBoardS]

BB_CORNERS = BB_A1 | BB_H1 | BB_A8 | BB_H8
BB_CENTER = BB_D4 | BB_E4 | BB_D5 | BB_E5

BB_LIGHT_SquareBoardS = 0x55aa_55aa_55aa_55aa
BB_DARK_SquareBoardS = 0xaa55_aa55_aa55_aa55

bitBoardFiles = [
    FileA,
    FileB,
    FileC,
    FileD,
    FileE,
    FileF,
    FileG,
    FileH,
] = [0x0101_0101_0101_0101 << i for i in range(8)]

bitBoardRanks = [
    Rank1,
    Rank2,
    Rank3,
    Rank4,
    Rank5,
    Rank6,
    Rank7,
    Rank8,
] = [0xff << (8 * i) for i in range(8)]

BB_BACKRANKS = Rank1 | Rank8


def lsb(bb: BitBoard) -> int:
    return (bb & -bb).bit_length() - 1


def scan_forward(bb: BitBoard) -> Iterator[SquareBoard]:
    while bb:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r


def msb(bb: BitBoard) -> int:
    return bb.bit_length() - 1


def scan_reversed(bb: BitBoard) -> Iterator[SquareBoard]:
    while bb:
        r = bb.bit_length() - 1
        yield r
        bb ^= BB_SquareBoardS[r]


# Python 3.10 or fallback.
popcount: Callable[[BitBoard], int] = getattr(
    int, "bit_count", lambda bb: bin(bb).count("1"))


def flip_vertical(bb: BitBoard) -> BitBoard:
    bb = ((bb >> 8) & 0x00ff_00ff_00ff_00ff) | (
        (bb & 0x00ff_00ff_00ff_00ff) << 8)
    bb = ((bb >> 16) & 0x0000_ffff_0000_ffff) | (
        (bb & 0x0000_ffff_0000_ffff) << 16)
    bb = (bb >> 32) | ((bb & 0x0000_0000_ffff_ffff) << 32)
    return bb


def flip_horizontal(bb: BitBoard) -> BitBoard:
    bb = ((bb >> 1) & 0x5555_5555_5555_5555) | (
        (bb & 0x5555_5555_5555_5555) << 1)
    bb = ((bb >> 2) & 0x3333_3333_3333_3333) | (
        (bb & 0x3333_3333_3333_3333) << 2)
    bb = ((bb >> 4) & 0x0f0f_0f0f_0f0f_0f0f) | (
        (bb & 0x0f0f_0f0f_0f0f_0f0f) << 4)
    return bb


def flip_diagonal(bb: BitBoard) -> BitBoard:
    t = (bb ^ (bb << 28)) & 0x0f0f_0f0f_0000_0000
    bb = bb ^ t ^ (t >> 28)
    t = (bb ^ (bb << 14)) & 0x3333_0000_3333_0000
    bb = bb ^ t ^ (t >> 14)
    t = (bb ^ (bb << 7)) & 0x5500_5500_5500_5500
    bb = bb ^ t ^ (t >> 7)
    return bb


def flip_anti_diagonal(bb: BitBoard) -> BitBoard:
    t = bb ^ (bb << 36)
    bb = bb ^ ((t ^ (bb >> 36)) & 0xf0f0_f0f0_0f0f_0f0f)
    t = (bb ^ (bb << 18)) & 0xcccc_0000_cccc_0000
    bb = bb ^ t ^ (t >> 18)
    t = (bb ^ (bb << 9)) & 0xaa00_aa00_aa00_aa00
    bb = bb ^ t ^ (t >> 9)
    return bb


def shift_down(b: BitBoard) -> BitBoard:
    return b >> 8


def shift_2_down(b: BitBoard) -> BitBoard:
    return b >> 16


def shift_up(b: BitBoard) -> BitBoard:
    return (b << 8) & BB_ALL


def shift_2_up(b: BitBoard) -> BitBoard:
    return (b << 16) & BB_ALL


def shift_right(b: BitBoard) -> BitBoard:
    return (b << 1) & ~FileA & BB_ALL


def shift_2_right(b: BitBoard) -> BitBoard:
    return (b << 2) & ~FileA & ~BB_FILE_B & BB_ALL


def shift_left(b: BitBoard) -> BitBoard:
    return (b >> 1) & ~FileH


def shift_2_left(b: BitBoard) -> BitBoard:
    return (b >> 2) & ~FileG & ~FileH


def shift_up_left(b: BitBoard) -> BitBoard:
    return (b << 7) & ~FileH & BB_ALL


def shift_up_right(b: BitBoard) -> BitBoard:
    return (b << 9) & ~FileA & BB_ALL


def shift_down_left(b: BitBoard) -> BitBoard:
    return (b >> 9) & ~FileH


def shift_down_right(b: BitBoard) -> BitBoard:
    return (b >> 7) & ~FileA


def _sliding_attacks(SquareBoard: SquareBoard, occupied: BitBoard, deltas: Iterable[int]) -> BitBoard:
    attacks = BB_EMPTY

    for delta in deltas:
        sq = SquareBoard

        while True:
            sq += delta
            if not (0 <= sq < 64) or SquareBoard_distance(sq, sq - delta) > 2:
                break

            attacks |= BB_SquareBoardS[sq]

            if occupied & BB_SquareBoardS[sq]:
                break

    return attacks


def _step_attacks(SquareBoard: SquareBoard, deltas: Iterable[int]) -> BitBoard:
    return _sliding_attacks(SquareBoard, BB_ALL, deltas)


BB_KNIGHT_ATTACKS = [_step_attacks(
    sq, [17, 15, 10, 6, -17, -15, -10, -6]) for sq in SquareBoardS]
BB_KING_ATTACKS = [_step_attacks(
    sq, [9, 8, 7, 1, -9, -8, -7, -1]) for sq in SquareBoardS]
BB_PAWN_ATTACKS = [[_step_attacks(sq, deltas) for sq in SquareBoardS]
                   for deltas in [[-7, -9], [7, 9]]]


def _edges(SquareBoard: SquareBoard) -> BitBoard:
    return (((Rank1 | Rank8) & ~bitBoardRanks[SquareBoard_rank(SquareBoard)]) |
            ((FileA | FileH) & ~bitBoardFiles[SquareBoard_file(SquareBoard)]))


def _carry_rippler(mask: BitBoard) -> Iterator[BitBoard]:
    subset = BB_EMPTY
    while True:
        yield subset
        subset = (subset - mask) & mask
        if not subset:
            break


def _attack_table(deltas: List[int]) -> Tuple[List[BitBoard], List[Dict[BitBoard, BitBoard]]]:
    mask_table = []
    attack_table = []

    for SquareBoard in SquareBoardS:
        attacks = {}

        mask = _sliding_attacks(SquareBoard, 0, deltas) & ~_edges(SquareBoard)
        for subset in _carry_rippler(mask):
            attacks[subset] = _sliding_attacks(SquareBoard, subset, deltas)

        attack_table.append(attacks)
        mask_table.append(mask)

    return mask_table, attack_table


BB_DIAG_MASKS, BB_DIAG_ATTACKS = _attack_table([-9, -7, 7, 9])
BB_FILE_MASKS, FileATTACKS = _attack_table([-8, 8])
BB_RANK_MASKS, BB_RANK_ATTACKS = _attack_table([-1, 1])


def _rays() -> List[List[BitBoard]]:
    rays = []
    for a, bb_a in enumerate(BB_SquareBoardS):
        rays_row = []
        for b, bb_b in enumerate(BB_SquareBoardS):
            if BB_DIAG_ATTACKS[a][0] & bb_b:
                rays_row.append(
                    (BB_DIAG_ATTACKS[a][0] & BB_DIAG_ATTACKS[b][0]) | bb_a | bb_b)
            elif BB_RANK_ATTACKS[a][0] & bb_b:
                rays_row.append(BB_RANK_ATTACKS[a][0] | bb_a)
            elif FileATTACKS[a][0] & bb_b:
                rays_row.append(FileATTACKS[a][0] | bb_a)
            else:
                rays_row.append(BB_EMPTY)
        rays.append(rays_row)
    return rays


BB_RAYS = _rays()


def ray(a: SquareBoard, b: SquareBoard) -> BitBoard:
    return BB_RAYS[a][b]


def between(a: SquareBoard, b: SquareBoard) -> BitBoard:
    bb = BB_RAYS[a][b] & ((BB_ALL << a) ^ (BB_ALL << b))
    return bb & (bb - 1)


SAN_REGEX = re.compile(
    r"^([NBKRQ])?([a-h])?([1-8])?[\-x]?([a-h][1-8])(=?[nbrqkNBRQK])?[\+#]?\Z")

FEN_CASTLING_REGEX = re.compile(r"^(?:-|[KQABCDEFGH]{0,2}[kqabcdefgh]{0,2})\Z")


@dataclasses.dataclass
class Piece:
    Piecetype: PieceType
    color: Color

    def symbol(self) -> str:
        symbol = pieceSymbol(self.Piecetype)
        return symbol.upper() if self.color else symbol

    def unicode_symbol(self, *, invert_color: bool = False) -> str:
        symbol = self.symbol().swapcase() if invert_color else self.symbol()
        return asci_dict[symbol]

    def __hash__(self) -> int:
        return self.Piecetype + (-1 if self.color else 5)

    def __repr__(self) -> str:
        return f"Piece.from_symbol({self.symbol()!r})"

    def __str__(self) -> str:
        return self.symbol()


    @classmethod
    def from_symbol(cls, symbol: str) -> Piece:
        return cls(pieceSymbolS.index(symbol.lower()), symbol.isupper())


@dataclasses.dataclass(unsafe_hash=True)
class Move:
    from_SquareBoard: SquareBoard
    to_SquareBoard: SquareBoard
    toQueen: Optional[PieceType] = None
    drop: Optional[PieceType] = None

    def uci(self) -> str:
        if self.drop:
            return pieceSymbol(self.drop).upper() + "@" + SquareBoardName[self.to_SquareBoard]
        elif self.toQueen:
            return SquareBoardName[self.from_SquareBoard] + SquareBoardName[self.to_SquareBoard] + pieceSymbol(self.toQueen)
        elif self:
            return SquareBoardName[self.from_SquareBoard] + SquareBoardName[self.to_SquareBoard]
        else:
            return "0000"

    def xboard(self) -> str:
        return self.uci() if self else "@@@@"

    def __bool__(self) -> bool:
        return bool(self.from_SquareBoard or self.to_SquareBoard or self.toQueen or self.drop)

    def __repr__(self) -> str:
        return f"Move.from_uci({self.uci()!r})"

    def __str__(self) -> str:
        return self.uci()

    @classmethod
    def from_uci(cls, uci: str) -> Move:
        if uci == "0000":
            return cls.null()
        elif len(uci) == 4 and "@" == uci[1]:
            try:
                drop = pieceSymbolS.index(uci[0].lower())
                SquareBoard = SquareBoardName.index(uci[2:])
            except ValueError:
                raise InvalidMoveError(f"invalid uci: {uci!r}")
            return cls(SquareBoard, SquareBoard, drop=drop)
        elif 4 <= len(uci) <= 5:
            try:
                from_SquareBoard = SquareBoardName.index(uci[0:2])
                to_SquareBoard = SquareBoardName.index(uci[2:4])
                toQueen = pieceSymbolS.index(
                    uci[4]) if len(uci) == 5 else None
            except ValueError:
                raise InvalidMoveError(f"invalid uci: {uci!r}")
            if from_SquareBoard == to_SquareBoard:
                raise InvalidMoveError(
                    f"invalid uci (use 0000 for null moves): {uci!r}")
            return cls(from_SquareBoard, to_SquareBoard, toQueen=toQueen)
        else:
            raise InvalidMoveError(
                f"expected uci string to be of length 4 or 5: {uci!r}")

    @classmethod
    def null(cls) -> Move:
        return cls(0, 0)


BaseBoardT = TypeVar("BaseBoardT", bound="BaseBoard")


class BaseBoard:

    def __init__(self, board_fen: Optional[str] = fenBoard) -> None:
        self.occupied_co = [BB_EMPTY, BB_EMPTY]

        if board_fen is None:
            self._clear_board()
        elif board_fen == fenBoard:
            self._reset_board()
        else:
            self._set_board_fen(board_fen)

    def _reset_board(self) -> None:
        self.pawns = Rank2 | Rank7
        self.knights = BB_B1 | BB_G1 | BB_B8 | BB_G8
        self.bishops = BB_C1 | BB_F1 | BB_C8 | BB_F8
        self.rooks = BB_CORNERS
        self.queens = BB_D1 | BB_D8
        self.kings = BB_E1 | BB_E8

        self.promoted = BB_EMPTY

        self.occupied_co[WHITE] = Rank1 | Rank2
        self.occupied_co[BLACK] = Rank7 | Rank8
        self.occupied = Rank1 | Rank2 | Rank7 | Rank8

    def reset_board(self) -> None:
        self._reset_board()

    def _clear_board(self) -> None:
        self.pawns = BB_EMPTY
        self.knights = BB_EMPTY
        self.bishops = BB_EMPTY
        self.rooks = BB_EMPTY
        self.queens = BB_EMPTY
        self.kings = BB_EMPTY

        self.promoted = BB_EMPTY

        self.occupied_co[WHITE] = BB_EMPTY
        self.occupied_co[BLACK] = BB_EMPTY
        self.occupied = BB_EMPTY

    def clear_board(self) -> None:
        self._clear_board()

    def pieces_mask(self, Piecetype: PieceType, color: Color) -> BitBoard:
        if Piecetype == PAWN:
            bb = self.pawns
        elif Piecetype == KNIGHT:
            bb = self.knights
        elif Piecetype == BISHOP:
            bb = self.bishops
        elif Piecetype == ROOK:
            bb = self.rooks
        elif Piecetype == QUEEN:
            bb = self.queens
        elif Piecetype == KING:
            bb = self.kings
        else:
            assert False, f"expected PieceType, got {Piecetype!r}"

        return bb & self.occupied_co[color]

    def piece_at(self, SquareBoard: SquareBoard) -> Optional[Piece]:
        Piecetype = self.Piecetype_at(SquareBoard)
        if Piecetype:
            mask = BB_SquareBoardS[SquareBoard]
            color = bool(self.occupied_co[WHITE] & mask)
            return Piece(Piecetype, color)
        else:
            return None

    def Piecetype_at(self, SquareBoard: SquareBoard) -> Optional[PieceType]:
        mask = BB_SquareBoardS[SquareBoard]

        if not self.occupied & mask:
            return None  # Early return
        elif self.pawns & mask:
            return PAWN
        elif self.knights & mask:
            return KNIGHT
        elif self.bishops & mask:
            return BISHOP
        elif self.rooks & mask:
            return ROOK
        elif self.queens & mask:
            return QUEEN
        else:
            return KING

    def color_at(self, SquareBoard: SquareBoard) -> Optional[Color]:
        mask = BB_SquareBoardS[SquareBoard]
        if self.occupied_co[WHITE] & mask:
            return WHITE
        elif self.occupied_co[BLACK] & mask:
            return BLACK
        else:
            return None

    def king(self, color: Color) -> Optional[SquareBoard]:
        king_mask = self.occupied_co[color] & self.kings & ~self.promoted
        return msb(king_mask) if king_mask else None

    def attacks_mask(self, SquareBoard: SquareBoard) -> BitBoard:
        bb_SquareBoard = BB_SquareBoardS[SquareBoard]

        if bb_SquareBoard & self.pawns:
            color = bool(bb_SquareBoard & self.occupied_co[WHITE])
            return BB_PAWN_ATTACKS[color][SquareBoard]
        elif bb_SquareBoard & self.knights:
            return BB_KNIGHT_ATTACKS[SquareBoard]
        elif bb_SquareBoard & self.kings:
            return BB_KING_ATTACKS[SquareBoard]
        else:
            attacks = 0
            if bb_SquareBoard & self.bishops or bb_SquareBoard & self.queens:
                attacks = BB_DIAG_ATTACKS[SquareBoard][BB_DIAG_MASKS[SquareBoard]
                                                  & self.occupied]
            if bb_SquareBoard & self.rooks or bb_SquareBoard & self.queens:
                attacks |= (BB_RANK_ATTACKS[SquareBoard][BB_RANK_MASKS[SquareBoard] & self.occupied] |
                            FileATTACKS[SquareBoard][BB_FILE_MASKS[SquareBoard] & self.occupied])
            return attacks



    def _attackers_mask(self, color: Color, SquareBoard: SquareBoard, occupied: BitBoard) -> BitBoard:
        rank_pieces = BB_RANK_MASKS[SquareBoard] & occupied
        file_pieces = BB_FILE_MASKS[SquareBoard] & occupied
        diag_pieces = BB_DIAG_MASKS[SquareBoard] & occupied

        queens_and_rooks = self.queens | self.rooks
        queens_and_bishops = self.queens | self.bishops

        attackers = (
            (BB_KING_ATTACKS[SquareBoard] & self.kings) |
            (BB_KNIGHT_ATTACKS[SquareBoard] & self.knights) |
            (BB_RANK_ATTACKS[SquareBoard][rank_pieces] & queens_and_rooks) |
            (FileATTACKS[SquareBoard][file_pieces] & queens_and_rooks) |
            (BB_DIAG_ATTACKS[SquareBoard][diag_pieces] & queens_and_bishops) |
            (BB_PAWN_ATTACKS[not color][SquareBoard] & self.pawns))

        return attackers & self.occupied_co[color]

    def attackers_mask(self, color: Color, SquareBoard: SquareBoard) -> BitBoard:
        return self._attackers_mask(color, SquareBoard, self.occupied)

    def is_attacked_by(self, color: Color, SquareBoard: SquareBoard) -> bool:
        return bool(self.attackers_mask(color, SquareBoard))

    def _remove_piece_at(self, SquareBoard: SquareBoard) -> Optional[PieceType]:
        Piecetype = self.Piecetype_at(SquareBoard)
        mask = BB_SquareBoardS[SquareBoard]

        if Piecetype == PAWN:
            self.pawns ^= mask
        elif Piecetype == KNIGHT:
            self.knights ^= mask
        elif Piecetype == BISHOP:
            self.bishops ^= mask
        elif Piecetype == ROOK:
            self.rooks ^= mask
        elif Piecetype == QUEEN:
            self.queens ^= mask
        elif Piecetype == KING:
            self.kings ^= mask
        else:
            return None

        self.occupied ^= mask
        self.occupied_co[WHITE] &= ~mask
        self.occupied_co[BLACK] &= ~mask

        self.promoted &= ~mask

        return Piecetype

    def remove_piece_at(self, SquareBoard: SquareBoard) -> Optional[Piece]:
        color = bool(self.occupied_co[WHITE] & BB_SquareBoardS[SquareBoard])
        Piecetype = self._remove_piece_at(SquareBoard)
        return Piece(Piecetype, color) if Piecetype else None

    def _set_piece_at(self, SquareBoard: SquareBoard, Piecetype: PieceType, color: Color, promoted: bool = False) -> None:
        self._remove_piece_at(SquareBoard)

        mask = BB_SquareBoardS[SquareBoard]

        if Piecetype == PAWN:
            self.pawns |= mask
        elif Piecetype == KNIGHT:
            self.knights |= mask
        elif Piecetype == BISHOP:
            self.bishops |= mask
        elif Piecetype == ROOK:
            self.rooks |= mask
        elif Piecetype == QUEEN:
            self.queens |= mask
        elif Piecetype == KING:
            self.kings |= mask
        else:
            return

        self.occupied ^= mask
        self.occupied_co[color] ^= mask

        if promoted:
            self.promoted ^= mask

    def set_piece_at(self, SquareBoard: SquareBoard, piece: Optional[Piece], promoted: bool = False) -> None:
        if piece is None:
            self._remove_piece_at(SquareBoard)
        else:
            self._set_piece_at(SquareBoard, piece.Piecetype, piece.color, promoted)

    def board_fen(self, *, promoted: Optional[bool] = False) -> str:
        builder = []
        empty = 0

        for SquareBoard in SquareBoardS_180:
            piece = self.piece_at(SquareBoard)

            if not piece:
                empty += 1
            else:
                if empty:
                    builder.append(str(empty))
                    empty = 0
                builder.append(piece.symbol())
                if promoted and BB_SquareBoardS[SquareBoard] & self.promoted:
                    builder.append("~")

            if BB_SquareBoardS[SquareBoard] & FileH:
                if empty:
                    builder.append(str(empty))
                    empty = 0

                if SquareBoard != H1:
                    builder.append("/")

        return "".join(builder)

    def _set_board_fen(self, fen: str) -> None:
        # Compatibility with set_fen().
        fen = fen.strip()
        if " " in fen:
            raise ValueError(
                f"expected position part of fen, got multiple parts: {fen!r}")

        # Ensure the FEN is valid.
        rows = fen.split("/")
        if len(rows) != 8:
            raise ValueError(
                f"expected 8 rows in position part of fen: {fen!r}")

        # Validate each row.
        for row in rows:
            field_sum = 0
            previous_was_digit = False
            previous_was_piece = False

            for c in row:
                if c in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                    if previous_was_digit:
                        raise ValueError(
                            f"two subsequent digits in position part of fen: {fen!r}")
                    field_sum += int(c)
                    previous_was_digit = True
                    previous_was_piece = False
                elif c == "~":
                    if not previous_was_piece:
                        raise ValueError(
                            f"'~' not after piece in position part of fen: {fen!r}")
                    previous_was_digit = False
                    previous_was_piece = False
                elif c.lower() in pieceSymbolS:
                    field_sum += 1
                    previous_was_digit = False
                    previous_was_piece = True
                else:
                    raise ValueError(
                        f"invalid character in position part of fen: {fen!r}")

            if field_sum != 8:
                raise ValueError(
                    f"expected 8 columns per row in position part of fen: {fen!r}")

        # Clear the board.
        self._clear_board()

        # Put pieces on the board.
        SquareBoard_index = 0
        for c in fen:
            if c in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                SquareBoard_index += int(c)
            elif c.lower() in pieceSymbolS:
                piece = Piece.from_symbol(c)
                self._set_piece_at(
                    SquareBoardS_180[SquareBoard_index], piece.Piecetype, piece.color)
                SquareBoard_index += 1
            elif c == "~":
                self.promoted |= BB_SquareBoardS[SquareBoardS_180[SquareBoard_index - 1]]

    def set_board_fen(self, fen: str) -> None:
        self._set_board_fen(fen)

    def piece_map(self, *, mask: BitBoard = BB_ALL) -> Dict[SquareBoard, Piece]:
        result = {}
        for SquareBoard in scan_reversed(self.occupied & mask):
            result[SquareBoard] = typing.cast(Piece, self.piece_at(SquareBoard))
        return result

    def _set_piece_map(self, pieces: Mapping[SquareBoard, Piece]) -> None:
        self._clear_board()
        for SquareBoard, piece in pieces.items():
            self._set_piece_at(SquareBoard, piece.Piecetype, piece.color)

    def set_piece_map(self, pieces: Mapping[SquareBoard, Piece]) -> None:
        self._set_piece_map(pieces)



    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.board_fen()!r})"

    def __str__(self) -> str:
        builder = []

        for SquareBoard in SquareBoardS_180:
            piece = self.piece_at(SquareBoard)

            if piece:
                builder.append(piece.symbol())
            else:
                builder.append(".")

            if BB_SquareBoardS[SquareBoard] & FileH:
                if SquareBoard != H1:
                    builder.append("\n")
            else:
                builder.append(" ")

        return "".join(builder)

    def unicode(self, *, invert_color: bool = False, borders: bool = False, empty_SquareBoard: str = "⭘", orientation: Color = WHITE) -> str:
        builder = []
        for rank_index in (range(7, -1, -1) if orientation else range(8)):
            if borders:
                builder.append("  ")
                builder.append("-" * 17)
                builder.append("\n")

                builder.append(RankName[rank_index])
                builder.append(" ")

            for i, file_index in enumerate(range(8) if orientation else range(7, -1, -1)):
                SquareBoard_index = SquareBoard(file_index, rank_index)

                if borders:
                    builder.append("|")
                elif i > 0:
                    builder.append(" ")

                piece = self.piece_at(SquareBoard_index)

                if piece:
                    builder.append(piece.unicode_symbol(
                        invert_color=invert_color))
                else:
                    builder.append(empty_SquareBoard)

            if borders:
                builder.append("|")

            if borders or (rank_index > 0 if orientation else rank_index < 7):
                builder.append("\n")

        if borders:
            builder.append("  ")
            builder.append("-" * 17)
            builder.append("\n")
            letters = "a b c d e f g h" if orientation else "h g f e d c b a"
            builder.append("   " + letters)

        return "".join(builder)

    def __eq__(self, board: object) -> bool:
        if isinstance(board, BaseBoard):
            return (
                self.occupied == board.occupied and
                self.occupied_co[WHITE] == board.occupied_co[WHITE] and
                self.pawns == board.pawns and
                self.knights == board.knights and
                self.bishops == board.bishops and
                self.rooks == board.rooks and
                self.queens == board.queens and
                self.kings == board.kings)
        else:
            return NotImplemented

    def apply_transform(self, f: Callable[[BitBoard], BitBoard]) -> None:
        self.pawns = f(self.pawns)
        self.knights = f(self.knights)
        self.bishops = f(self.bishops)
        self.rooks = f(self.rooks)
        self.queens = f(self.queens)
        self.kings = f(self.kings)

        self.occupied_co[WHITE] = f(self.occupied_co[WHITE])
        self.occupied_co[BLACK] = f(self.occupied_co[BLACK])
        self.occupied = f(self.occupied)
        self.promoted = f(self.promoted)

    def transform(self: BaseBoardT, f: Callable[[BitBoard], BitBoard]) -> BaseBoardT:
        board = self.copy()
        board.apply_transform(f)
        return board

    def apply_mirror(self: BaseBoardT) -> None:
        self.apply_transform(flip_vertical)
        self.occupied_co[WHITE], self.occupied_co[BLACK] = self.occupied_co[BLACK], self.occupied_co[WHITE]

    def mirror(self: BaseBoardT) -> BaseBoardT:
        board = self.copy()
        board.apply_mirror()
        return board

    def copy(self: BaseBoardT) -> BaseBoardT:
        """Creates a copy of the board."""
        board = type(self)(None)

        board.pawns = self.pawns
        board.knights = self.knights
        board.bishops = self.bishops
        board.rooks = self.rooks
        board.queens = self.queens
        board.kings = self.kings

        board.occupied_co[WHITE] = self.occupied_co[WHITE]
        board.occupied_co[BLACK] = self.occupied_co[BLACK]
        board.occupied = self.occupied
        board.promoted = self.promoted

        return board

    def __copy__(self: BaseBoardT) -> BaseBoardT:
        return self.copy()

    def __deepcopy__(self: BaseBoardT, memo: Dict[int, object]) -> BaseBoardT:
        board = self.copy()
        memo[id(self)] = board
        return board

    @classmethod
    def empty(cls: Type[BaseBoardT]) -> BaseBoardT:
        return cls(None)




BoardT = TypeVar("BoardT", bound="Board")


class _BoardState(Generic[BoardT]):

    def __init__(self, board: BoardT) -> None:
        self.pawns = board.pawns
        self.knights = board.knights
        self.bishops = board.bishops
        self.rooks = board.rooks
        self.queens = board.queens
        self.kings = board.kings

        self.occupied_w = board.occupied_co[WHITE]
        self.occupied_b = board.occupied_co[BLACK]
        self.occupied = board.occupied

        self.promoted = board.promoted

        self.turn = board.turn
        self.castling_rights = board.castling_rights
        self.ep_SquareBoard = board.ep_SquareBoard
        self.halfmove_clock = board.halfmove_clock
        self.fmoveNum = board.fmoveNum

    def restore(self, board: BoardT) -> None:
        board.pawns = self.pawns
        board.knights = self.knights
        board.bishops = self.bishops
        board.rooks = self.rooks
        board.queens = self.queens
        board.kings = self.kings

        board.occupied_co[WHITE] = self.occupied_w
        board.occupied_co[BLACK] = self.occupied_b
        board.occupied = self.occupied

        board.promoted = self.promoted

        board.turn = self.turn
        board.castling_rights = self.castling_rights
        board.ep_SquareBoard = self.ep_SquareBoard
        board.halfmove_clock = self.halfmove_clock
        board.fmoveNum = self.fmoveNum


class Board(BaseBoard):

    aliases: ClassVar[List[str]] = ["Standard", "Chess",
                                    "Classical", "Normal", "Illegal", "From Position"]
    uci_variant: ClassVar[Optional[str]] = "chess"
    xboard_variant: ClassVar[Optional[str]] = "normal"
    fenStart: ClassVar[str] = fenStart

    tbw_suffix: ClassVar[Optional[str]] = ".rtbw"
    tbz_suffix: ClassVar[Optional[str]] = ".rtbz"
    tbw_magic: ClassVar[Optional[bytes]] = b"\x71\xe8\x23\x5d"
    tbz_magic: ClassVar[Optional[bytes]] = b"\xd7\x66\x0c\xa5"
    pawnless_tbw_suffix: ClassVar[Optional[str]] = None
    pawnless_tbz_suffix: ClassVar[Optional[str]] = None
    pawnless_tbw_magic: ClassVar[Optional[bytes]] = None
    pawnless_tbz_magic: ClassVar[Optional[bytes]] = None
    connected_kings: ClassVar[bool] = False
    one_king: ClassVar[bool] = True
    captures_compulsory: ClassVar[bool] = False

    turn: Color
    castling_rights: BitBoard
    ep_SquareBoard: Optional[SquareBoard]
    fmoveNum: int
    halfmove_clock: int
    promoted: BitBoard
    chess960: bool

    move_stack: List[Move]

    def __init__(self: BoardT, fen: Optional[str] = fenStart, *, chess960: bool = False) -> None:
        BaseBoard.__init__(self, None)

        self.chess960 = chess960

        self.ep_SquareBoard = None
        self.move_stack = []
        self._stack: List[_BoardState[BoardT]] = []

        if fen is None:
            self.clear()
        elif fen == type(self).fenStart:
            self.reset()
        else:
            self.set_fen(fen)

    @property
    def legal_moves(self) -> LegalMoveGenerator:
        return LegalMoveGenerator(self)

    @property
    def pseudo_legal_moves(self) -> PseudoLegalMoveGenerator:
        return PseudoLegalMoveGenerator(self)

    def reset(self) -> None:
        self.turn = WHITE
        self.castling_rights = BB_CORNERS
        self.ep_SquareBoard = None
        self.halfmove_clock = 0
        self.fmoveNum = 1

        self.reset_board()

    def reset_board(self) -> None:
        super().reset_board()
        self.clear_stack()

    def clear(self) -> None:
        self.turn = WHITE
        self.castling_rights = BB_EMPTY
        self.ep_SquareBoard = None
        self.halfmove_clock = 0
        self.fmoveNum = 1

        self.clear_board()

    def clear_board(self) -> None:
        super().clear_board()
        self.clear_stack()

    def clear_stack(self) -> None:
        self.move_stack.clear()
        self._stack.clear()

    def root(self: BoardT) -> BoardT:
        if self._stack:
            board = type(self)(None, chess960=self.chess960)
            self._stack[0].restore(board)
            return board
        else:
            return self.copy(stack=False)

    def ply(self) -> int:
        return 2 * (self.fmoveNum - 1) + (self.turn == BLACK)

    def remove_piece_at(self, SquareBoard: SquareBoard) -> Optional[Piece]:
        piece = super().remove_piece_at(SquareBoard)
        self.clear_stack()
        return piece

    def set_piece_at(self, SquareBoard: SquareBoard, piece: Optional[Piece], promoted: bool = False) -> None:
        super().set_piece_at(SquareBoard, piece, promoted=promoted)
        self.clear_stack()

    def generate_pseudo_legal_moves(self, maskIn: BitBoard = BB_ALL, to_mask: BitBoard = BB_ALL) -> Iterator[Move]:
        our_pieces = self.occupied_co[self.turn]

        # Generate piece moves.
        non_pawns = our_pieces & ~self.pawns & maskIn
        for from_SquareBoard in scan_reversed(non_pawns):
            moves = self.attacks_mask(from_SquareBoard) & ~our_pieces & to_mask
            for to_SquareBoard in scan_reversed(moves):
                yield Move(from_SquareBoard, to_SquareBoard)

        # Generate castling moves.
        if maskIn & self.kings:
            yield from self.generate_castling_moves(maskIn, to_mask)

        # The remaining moves are all pawn moves.
        pawns = self.pawns & self.occupied_co[self.turn] & maskIn
        if not pawns:
            return

        # Generate pawn captures.
        capturers = pawns
        for from_SquareBoard in scan_reversed(capturers):
            targets = (
                BB_PAWN_ATTACKS[self.turn][from_SquareBoard] &
                self.occupied_co[not self.turn] & to_mask)

            for to_SquareBoard in scan_reversed(targets):
                if SquareBoard_rank(to_SquareBoard) in [0, 7]:
                    yield Move(from_SquareBoard, to_SquareBoard, QUEEN)
                    yield Move(from_SquareBoard, to_SquareBoard, ROOK)
                    yield Move(from_SquareBoard, to_SquareBoard, BISHOP)
                    yield Move(from_SquareBoard, to_SquareBoard, KNIGHT)
                else:
                    yield Move(from_SquareBoard, to_SquareBoard)

        # Prepare pawn advance generation.
        if self.turn == WHITE:
            single_moves = pawns << 8 & ~self.occupied
            double_moves = single_moves << 8 & ~self.occupied & (
                Rank3 | Rank4)
        else:
            single_moves = pawns >> 8 & ~self.occupied
            double_moves = single_moves >> 8 & ~self.occupied & (
                Rank6 | Rank5)

        single_moves &= to_mask
        double_moves &= to_mask

        # Generate single pawn moves.
        for to_SquareBoard in scan_reversed(single_moves):
            from_SquareBoard = to_SquareBoard + (8 if self.turn == BLACK else -8)

            if SquareBoard_rank(to_SquareBoard) in [0, 7]:
                yield Move(from_SquareBoard, to_SquareBoard, QUEEN)
                yield Move(from_SquareBoard, to_SquareBoard, ROOK)
                yield Move(from_SquareBoard, to_SquareBoard, BISHOP)
                yield Move(from_SquareBoard, to_SquareBoard, KNIGHT)
            else:
                yield Move(from_SquareBoard, to_SquareBoard)

        # Generate double pawn moves.
        for to_SquareBoard in scan_reversed(double_moves):
            from_SquareBoard = to_SquareBoard + (16 if self.turn == BLACK else -16)
            yield Move(from_SquareBoard, to_SquareBoard)

        # Generate en passant captures.
        if self.ep_SquareBoard:
            yield from self.generate_pseudo_legal_ep(maskIn, to_mask)

    def generate_pseudo_legal_ep(self, maskIn: BitBoard = BB_ALL, to_mask: BitBoard = BB_ALL) -> Iterator[Move]:
        if not self.ep_SquareBoard or not BB_SquareBoardS[self.ep_SquareBoard] & to_mask:
            return

        if BB_SquareBoardS[self.ep_SquareBoard] & self.occupied:
            return

        capturers = (
            self.pawns & self.occupied_co[self.turn] & maskIn &
            BB_PAWN_ATTACKS[not self.turn][self.ep_SquareBoard] &
            bitBoardRanks[4 if self.turn else 3])

        for capturer in scan_reversed(capturers):
            yield Move(capturer, self.ep_SquareBoard)

    def generate_pseudo_legal_captures(self, maskIn: BitBoard = BB_ALL, to_mask: BitBoard = BB_ALL) -> Iterator[Move]:
        return itertools.chain(
            self.generate_pseudo_legal_moves(
                maskIn, to_mask & self.occupied_co[not self.turn]),
            self.generate_pseudo_legal_ep(maskIn, to_mask))

    def checkers_mask(self) -> BitBoard:
        king = self.king(self.turn)
        return BB_EMPTY if king is None else self.attackers_mask(not self.turn, king)


    def is_check(self) -> bool:
        return bool(self.checkers_mask())

    def gives_check(self, move: Move) -> bool:
        self.push(move)
        try:
            return self.is_check()
        finally:
            self.pop()

    def is_into_check(self, move: Move) -> bool:
        king = self.king(self.turn)
        if king is None:
            return False

        # If already in check, look if it is an evasion.
        checkers = self.attackers_mask(not self.turn, king)
        if checkers and move not in self._generate_evasions(king, checkers, BB_SquareBoardS[move.from_SquareBoard], BB_SquareBoardS[move.to_SquareBoard]):
            return True

        return not self._is_safe(king, self._slider_blockers(king), move)

    def was_into_check(self) -> bool:
        king = self.king(not self.turn)
        return king is not None and self.is_attacked_by(self.turn, king)

    def is_pseudo_legal(self, move: Move) -> bool:
        # Null moves are not pseudo-legal.
        if not move:
            return False

        # Drops are not pseudo-legal.
        if move.drop:
            return False

        # Source SquareBoard must not be vacant.
        piece = self.Piecetype_at(move.from_SquareBoard)
        if not piece:
            return False

        # Get SquareBoard masks.
        maskIn = BB_SquareBoardS[move.from_SquareBoard]
        to_mask = BB_SquareBoardS[move.to_SquareBoard]

        # Check turn.
        if not self.occupied_co[self.turn] & maskIn:
            return False

        # Only pawns can promote and only on the backrank.
        if move.toQueen:
            if piece != PAWN:
                return False

            if self.turn == WHITE and SquareBoard_rank(move.to_SquareBoard) != 7:
                return False
            elif self.turn == BLACK and SquareBoard_rank(move.to_SquareBoard) != 0:
                return False

        # Handle castling.
        if piece == KING:
            move = self._from_chess960(
                self.chess960, move.from_SquareBoard, move.to_SquareBoard)
            if move in self.generate_castling_moves():
                return True

        # Destination SquareBoard can not be occupied.
        if self.occupied_co[self.turn] & to_mask:
            return False

        # Handle pawn moves.
        if piece == PAWN:
            return move in self.generate_pseudo_legal_moves(maskIn, to_mask)

        # Handle all other pieces.
        return bool(self.attacks_mask(move.from_SquareBoard) & to_mask)

    def is_legal(self, move: Move) -> bool:
        return not self.is_variant_end() and self.is_pseudo_legal(move) and not self.is_into_check(move)

    def is_variant_end(self) -> bool:
        return False

    def is_variant_loss(self) -> bool:
        return False

    def is_variant_win(self) -> bool:
        return False

    def is_variant_draw(self) -> bool:
        return False

    def is_game_over(self, *, claim_draw: bool = False) -> bool:
        return self.outcome(claim_draw=claim_draw) is not None

    def result(self, *, claim_draw: bool = False) -> str:
        outcome = self.outcome(claim_draw=claim_draw)
        return outcome.result() if outcome else "*"

    def outcome(self, *, claim_draw: bool = False) -> Optional[Outcome]:
        # Variant support.
        if self.is_variant_loss():
            return Outcome(Termination.VARIANT_LOSS, not self.turn)
        if self.is_variant_win():
            return Outcome(Termination.VARIANT_WIN, self.turn)
        if self.is_variant_draw():
            return Outcome(Termination.VARIANT_DRAW, None)

        # Normal game end.
        if self.is_checkmate():
            return Outcome(Termination.CHECKMATE, not self.turn)
        if self.is_insufficient_material():
            return Outcome(Termination.INSUFFICIENT_MATERIAL, None)
        if not any(self.generate_legal_moves()):
            return Outcome(Termination.STALEMATE, None)

        # Automatic draws.
        if self.is_seventyfive_moves():
            return Outcome(Termination.SEVENTYFIVE_MOVES, None)
        if self.is_fivefold_repetition():
            return Outcome(Termination.FIVEFOLD_REPETITION, None)

        # Claimable draws.
        if claim_draw:
            if self.can_claim_fifty_moves():
                return Outcome(Termination.FIFTY_MOVES, None)
            if self.can_claim_threefold_repetition():
                return Outcome(Termination.THREEFOLD_REPETITION, None)

        return None

    def is_checkmate(self) -> bool:
        if not self.is_check():
            return False

        return not any(self.generate_legal_moves())

    def is_stalemate(self) -> bool:
        if self.is_check():
            return False

        if self.is_variant_end():
            return False

        return not any(self.generate_legal_moves())

    def is_insufficient_material(self) -> bool:
        return all(self.has_insufficient_material(color) for color in COLORS)

    def has_insufficient_material(self, color: Color) -> bool:
        if self.occupied_co[color] & (self.pawns | self.rooks | self.queens):
            return False

        if self.occupied_co[color] & self.knights:
            return (popcount(self.occupied_co[color]) <= 2 and
                    not (self.occupied_co[not color] & ~self.kings & ~self.queens))

        if self.occupied_co[color] & self.bishops:
            same_color = (not self.bishops & BB_DARK_SquareBoardS) or (
                not self.bishops & BB_LIGHT_SquareBoardS)
            return same_color and not self.pawns and not self.knights

        return True

    def _is_halfmoves(self, n: int) -> bool:
        return self.halfmove_clock >= n and any(self.generate_legal_moves())

    def is_seventyfive_moves(self) -> bool:
        return self._is_halfmoves(150)

    def is_fivefold_repetition(self) -> bool:
        return self.is_repetition(5)

    def can_claim_draw(self) -> bool:
        return self.can_claim_fifty_moves() or self.can_claim_threefold_repetition()

    def is_fifty_moves(self) -> bool:
        return self._is_halfmoves(100)

    def can_claim_fifty_moves(self) -> bool:
        if self.is_fifty_moves():
            return True

        if self.halfmove_clock >= 99:
            for move in self.generate_legal_moves():
                if not self.is_zeroing(move):
                    self.push(move)
                    try:
                        if self.is_fifty_moves():
                            return True
                    finally:
                        self.pop()

        return False

    def can_claim_threefold_repetition(self) -> bool:
        transposition_key = self._transposition_key()
        transpositions: Counter[Hashable] = collections.Counter()
        transpositions.update((transposition_key, ))

        # Count positions.
        switchyard = []
        while self.move_stack:
            move = self.pop()
            switchyard.append(move)

            if self.is_irreversible(move):
                break

            transpositions.update((self._transposition_key(), ))

        while switchyard:
            self.push(switchyard.pop())

        # Threefold repetition occurred.
        if transpositions[transposition_key] >= 3:
            return True

        # The next legal move is a threefold repetition.
        for move in self.generate_legal_moves():
            self.push(move)
            try:
                if transpositions[self._transposition_key()] >= 2:
                    return True
            finally:
                self.pop()

        return False

    def is_repetition(self, count: int = 3) -> bool:
        maybe_repetitions = 1
        for state in reversed(self._stack):
            if state.occupied == self.occupied:
                maybe_repetitions += 1
                if maybe_repetitions >= count:
                    break
        if maybe_repetitions < count:
            return False


        transposition_key = self._transposition_key()
        switchyard = []

        try:
            while True:
                if count <= 1:
                    return True

                if len(self.move_stack) < count - 1:
                    break

                move = self.pop()
                switchyard.append(move)

                if self.is_irreversible(move):
                    break

                if self._transposition_key() == transposition_key:
                    count -= 1
        finally:
            while switchyard:
                self.push(switchyard.pop())

        return False

    def _board_state(self: BoardT) -> _BoardState[BoardT]:
        return _BoardState(self)

    def _push_capture(self, move: Move, capture_SquareBoard: SquareBoard, Piecetype: PieceType, was_promoted: bool) -> None:
        pass

    def push(self: BoardT, move: Move) -> None:
        # Push move and remember board state.
        move = self._to_chess960(move)
        board_state = self._board_state()
        self.castling_rights = self.clean_castling_rights()  # Before pushing stack
        self.move_stack.append(self._from_chess960(
            self.chess960, move.from_SquareBoard, move.to_SquareBoard, move.toQueen, move.drop))
        self._stack.append(board_state)

        # Reset en passant SquareBoard.
        ep_SquareBoard = self.ep_SquareBoard
        self.ep_SquareBoard = None

        # Increment move counters.
        self.halfmove_clock += 1
        if self.turn == BLACK:
            self.fmoveNum += 1

        # On a null move, simply swap turns and reset the en passant SquareBoard.
        if not move:
            self.turn = not self.turn
            return

        # Drops.
        if move.drop:
            self._set_piece_at(move.to_SquareBoard, move.drop, self.turn)
            self.turn = not self.turn
            return

        # Zero the half-move clock.
        if self.is_zeroing(move):
            self.halfmove_clock = 0

        from_bb = BB_SquareBoardS[move.from_SquareBoard]
        to_bb = BB_SquareBoardS[move.to_SquareBoard]

        promoted = bool(self.promoted & from_bb)
        Piecetype = self._remove_piece_at(move.from_SquareBoard)
        assert Piecetype is not None, f"push() expects move to be pseudo-legal, but got {move} in {self.board_fen()}"
        capture_SquareBoard = move.to_SquareBoard
        captured_Piecetype = self.Piecetype_at(capture_SquareBoard)

        # Update castling rights.
        self.castling_rights &= ~to_bb & ~from_bb
        if Piecetype == KING and not promoted:
            if self.turn == WHITE:
                self.castling_rights &= ~Rank1
            else:
                self.castling_rights &= ~Rank8
        elif captured_Piecetype == KING and not self.promoted & to_bb:
            if self.turn == WHITE and SquareBoard_rank(move.to_SquareBoard) == 7:
                self.castling_rights &= ~Rank8
            elif self.turn == BLACK and SquareBoard_rank(move.to_SquareBoard) == 0:
                self.castling_rights &= ~Rank1
 
        if Piecetype == PAWN:
            diff = move.to_SquareBoard - move.from_SquareBoard

            if diff == 16 and SquareBoard_rank(move.from_SquareBoard) == 1:
                self.ep_SquareBoard = move.from_SquareBoard + 8
            elif diff == -16 and SquareBoard_rank(move.from_SquareBoard) == 6:
                self.ep_SquareBoard = move.from_SquareBoard - 8
            elif move.to_SquareBoard == ep_SquareBoard and abs(diff) in [7, 9] and not captured_Piecetype:
                # Remove pawns captured en passant.
                down = -8 if self.turn == WHITE else 8
                capture_SquareBoard = ep_SquareBoard + down
                captured_Piecetype = self._remove_piece_at(capture_SquareBoard)
 
        if move.toQueen:
            promoted = True
            Piecetype = move.toQueen
 
        castling = Piecetype == KING and self.occupied_co[self.turn] & to_bb
        if castling:
            a_side = SquareBoard_file(
                move.to_SquareBoard) < SquareBoard_file(move.from_SquareBoard)

            self._remove_piece_at(move.from_SquareBoard)
            self._remove_piece_at(move.to_SquareBoard)

            if a_side:
                self._set_piece_at(C1 if self.turn ==
                                   WHITE else C8, KING, self.turn)
                self._set_piece_at(D1 if self.turn ==
                                   WHITE else D8, ROOK, self.turn)
            else:
                self._set_piece_at(G1 if self.turn ==
                                   WHITE else G8, KING, self.turn)
                self._set_piece_at(F1 if self.turn ==
                                   WHITE else F8, ROOK, self.turn)
 
        if not castling:
            was_promoted = bool(self.promoted & to_bb)
            self._set_piece_at(move.to_SquareBoard, Piecetype, self.turn, promoted)

            if captured_Piecetype:
                self._push_capture(move, capture_SquareBoard,
                                   captured_Piecetype, was_promoted)
 
        self.turn = not self.turn

    def pop(self: BoardT) -> Move:
        move = self.move_stack.pop()
        self._stack.pop().restore(self)
        return move

    def peek(self) -> Move:
        return self.move_stack[-1]

    def find_move(self, from_SquareBoard: SquareBoard, to_SquareBoard: SquareBoard, toQueen: Optional[PieceType] = None) -> Move:
        if toQueen is None and self.pawns & BB_SquareBoardS[from_SquareBoard] and BB_SquareBoardS[to_SquareBoard] & BB_BACKRANKS:
            toQueen = QUEEN

        move = self._from_chess960(
            self.chess960, from_SquareBoard, to_SquareBoard, toQueen)
        if not self.is_legal(move):
            raise IllegalMoveError(
                f"no matching legal move for {move.uci()} ({SquareBoardName[from_SquareBoard]} -> {SquareBoardName[to_SquareBoard]}) in {self.fen()}")

        return move

    def castling_shredder_fen(self) -> str:
        castling_rights = self.clean_castling_rights()
        if not castling_rights:
            return "-"

        builder = []

        for SquareBoard in scan_reversed(castling_rights & Rank1):
            builder.append(FileName[SquareBoard_file(SquareBoard)].upper())

        for SquareBoard in scan_reversed(castling_rights & Rank8):
            builder.append(FileName[SquareBoard_file(SquareBoard)])

        return "".join(builder)



    def has_pseudo_legal_en_passant(self) -> bool:
        return self.ep_SquareBoard is not None and any(self.generate_pseudo_legal_ep())

    def has_legal_en_passant(self) -> bool:
        return self.ep_SquareBoard is not None and any(self.generate_legal_ep())

    def fen(self, *, shredder: bool = False, en_passant: Passant = "legal", promoted: Optional[bool] = None) -> str:
        return " ".join([
            self.epd(shredder=shredder, en_passant=en_passant,
                     promoted=promoted),
            str(self.halfmove_clock),
            str(self.fmoveNum)
        ])

    def shredder_fen(self, *, en_passant: Passant = "legal", promoted: Optional[bool] = None) -> str:
        return " ".join([
            self.epd(shredder=True, en_passant=en_passant, promoted=promoted),
            str(self.halfmove_clock),
            str(self.fmoveNum)
        ])

    def set_fen(self, fen: str) -> None:
        parts = fen.split()

        try:
            board_part = parts.pop(0)
        except IndexError:
            raise ValueError("empty fen")
 
        try:
            turn_part = parts.pop(0)
        except IndexError:
            turn = WHITE
        else:
            if turn_part == "w":
                turn = WHITE
            elif turn_part == "b":
                turn = BLACK
            else:
                raise ValueError(
                    f"expected 'w' or 'b' for turn part of fen: {fen!r}")
 
        try:
            castling_part = parts.pop(0)
        except IndexError:
            castling_part = "-"
        else:
            if not FEN_CASTLING_REGEX.match(castling_part):
                raise ValueError(f"invalid castling part in fen: {fen!r}")
 
        try:
            ep_part = parts.pop(0)
        except IndexError:
            ep_SquareBoard = None
        else:
            try:
                ep_SquareBoard = None if ep_part == "-" else SquareBoardName.index(
                    ep_part)
            except ValueError:
                raise ValueError(f"invalid en passant part in fen: {fen!r}")

        # Check that the half-move part is valid.
        try:
            halfmove_part = parts.pop(0)
        except IndexError:
            halfmove_clock = 0
        else:
            try:
                halfmove_clock = int(halfmove_part)
            except ValueError:
                raise ValueError(f"invalid half-move clock in fen: {fen!r}")

            if halfmove_clock < 0:
                raise ValueError(
                    f"half-move clock cannot be negative: {fen!r}")

        # Check that the full-move number part is valid.
        # 0 is allowed for compatibility, but later replaced with 1.
        try:
            fullmove_part = parts.pop(0)
        except IndexError:
            fmoveNum = 1
        else:
            try:
                fmoveNum = int(fullmove_part)
            except ValueError:
                raise ValueError(f"invalid fullmove number in fen: {fen!r}")

            if fmoveNum < 0:
                raise ValueError(
                    f"fullmove number cannot be negative: {fen!r}")

            fmoveNum = max(fmoveNum, 1)

        if parts:
            raise ValueError(
                f"fen string has more parts than expected: {fen!r}")

        self._set_board_fen(board_part)


        self.turn = turn
        self._set_castling_fen(castling_part)
        self.ep_SquareBoard = ep_SquareBoard
        self.halfmove_clock = halfmove_clock
        self.fmoveNum = fmoveNum
        self.clear_stack()

    def _set_castling_fen(self, castling_fen: str) -> None:
        if not castling_fen or castling_fen == "-":
            self.castling_rights = BB_EMPTY
            return

        if not FEN_CASTLING_REGEX.match(castling_fen):
            raise ValueError(f"invalid castling fen: {castling_fen!r}")

        self.castling_rights = BB_EMPTY

        for flag in castling_fen:
            color = WHITE if flag.isupper() else BLACK
            flag = flag.lower()
            backrank = Rank1 if color == WHITE else Rank8
            rooks = self.occupied_co[color] & self.rooks & backrank
            king = self.king(color)

            if flag == "q":
                if king is not None and lsb(rooks) < king:
                    self.castling_rights |= rooks & -rooks
                else:
                    self.castling_rights |= FileA & backrank
            elif flag == "k":
                rook = msb(rooks)
                if king is not None and king < rook:
                    self.castling_rights |= BB_SquareBoardS[rook]
                else:
                    self.castling_rights |= FileH & backrank
            else:
                self.castling_rights |= bitBoardFiles[FileName.index(
                    flag)] & backrank

    def set_castling_fen(self, castling_fen: str) -> None:
        self._set_castling_fen(castling_fen)
        self.clear_stack()

    def set_board_fen(self, fen: str) -> None:
        super().set_board_fen(fen)
        self.clear_stack()

    def set_piece_map(self, pieces: Mapping[SquareBoard, Piece]) -> None:
        super().set_piece_map(pieces)
        self.clear_stack()
 

   
    def _epd_operations(self, operations: Mapping[str, Union[None, str, int, float, Move, Iterable[Move]]]) -> str:
        epd = []
        first_op = True

        for opcode, operand in operations.items():
            assert opcode != "-", "dash (-) is not a valid epd opcode"
            for blacklisted in [" ", "\n", "\t", "\r"]:
                assert blacklisted not in opcode, f"invalid character {blacklisted!r} in epd opcode: {opcode!r}"

            if not first_op:
                epd.append(" ")
            first_op = False
            epd.append(opcode)

            if operand is None:
                epd.append(";")
            elif isinstance(operand, Move):
                epd.append(" ")
                epd.append(self.san(operand))
                epd.append(";")
            elif isinstance(operand, int):
                epd.append(f" {operand};")
            elif isinstance(operand, float):
                assert math.isfinite(
                    operand), f"expected numeric epd operand to be finite, got: {operand}"
                epd.append(f" {operand};")
            elif opcode == "pv" and not isinstance(operand, str) and hasattr(operand, "__iter__"):
                position = self.copy(stack=False)
                for move in operand:
                    epd.append(" ")
                    epd.append(position.san_and_push(move))
                epd.append(";")
            elif opcode in ["am", "bm"] and not isinstance(operand, str) and hasattr(operand, "__iter__"):
                for san in sorted(self.san(move) for move in operand):
                    epd.append(" ")
                    epd.append(san)
                epd.append(";")
            else:
                # Append as escaped string.
                epd.append(" \"")
                epd.append(str(operand).replace("\\", "\\\\").replace("\t", "\\t").replace(
                    "\r", "\\r").replace("\n", "\\n").replace("\"", "\\\""))
                epd.append("\";")

        return "".join(epd)



    def san(self, move: Move) -> str:
        return self._algebraic(move)

    def lan(self, move: Move) -> str:
        return self._algebraic(move, long=True)

    def san_and_push(self, move: Move) -> str:
        return self._algebraic_and_push(move)

    def _algebraic(self, move: Move, *, long: bool = False) -> str:
        san = self._algebraic_and_push(move, long=long)
        self.pop()
        return san

    def _algebraic_and_push(self, move: Move, *, long: bool = False) -> str:
        san = self._algebraic_without_suffix(move, long=long)

        self.push(move)
        is_check = self.is_check()
        is_checkmate = (is_check and self.is_checkmate()
                        ) or self.is_variant_loss() or self.is_variant_win()

        if is_checkmate and move:
            return san + "#"
        elif is_check and move:
            return san + "+"
        else:
            return san

    def _algebraic_without_suffix(self, move: Move, *, long: bool = False) -> str:
        if not move:
            return "--"

        if move.drop:
            san = ""
            if move.drop != PAWN:
                san = pieceSymbol(move.drop).upper()
            san += "@" + SquareBoardName[move.to_SquareBoard]
            return san

        if self.is_castling(move):
            if SquareBoard_file(move.to_SquareBoard) < SquareBoard_file(move.from_SquareBoard):
                return "O-O-O"
            else:
                return "O-O"

        Piecetype = self.Piecetype_at(move.from_SquareBoard)
        assert Piecetype, f"san() and lan() expect move to be legal or null, but got {move} in {self.fen()}"
        capture = self.is_capture(move)

        if Piecetype == PAWN:
            san = ""
        else:
            san = pieceSymbol(Piecetype).upper()

        if long:
            san += SquareBoardName[move.from_SquareBoard]
        elif Piecetype != PAWN:
            others = 0
            maskIn = self.pieces_mask(Piecetype, self.turn)
            maskIn &= ~BB_SquareBoardS[move.from_SquareBoard]
            to_mask = BB_SquareBoardS[move.to_SquareBoard]
            for candidate in self.generate_legal_moves(maskIn, to_mask):
                others |= BB_SquareBoardS[candidate.from_SquareBoard]

            # Disambiguate.
            if others:
                row, column = False, False

                if others & bitBoardRanks[SquareBoard_rank(move.from_SquareBoard)]:
                    column = True

                if others & bitBoardFiles[SquareBoard_file(move.from_SquareBoard)]:
                    row = True
                else:
                    column = True

                if column:
                    san += FileName[SquareBoard_file(move.from_SquareBoard)]
                if row:
                    san += RankName[SquareBoard_rank(move.from_SquareBoard)]
        elif capture:
            san += FileName[SquareBoard_file(move.from_SquareBoard)]


        if capture:
            san += "x"
        elif long:
            san += "-"


        san += SquareBoardName[move.to_SquareBoard]


        if move.toQueen:
            san += "=" + pieceSymbol(move.toQueen).upper()

        return san



    def parse_san(self, san: str) -> Move:
        try:
            if san in ["O-O", "O-O+", "O-O#", "0-0", "0-0+", "0-0#"]:
                return next(move for move in self.generate_castling_moves() if self.is_kingside_castling(move))
            elif san in ["O-O-O", "O-O-O+", "O-O-O#", "0-0-0", "0-0-0+", "0-0-0#"]:
                return next(move for move in self.generate_castling_moves() if self.is_queenside_castling(move))
        except StopIteration:
            raise IllegalMoveError(f"illegal san: {san!r} in {self.fen()}")

        # Match normal moves.
        match = SAN_REGEX.match(san)
        if not match:
            # Null moves.
            if san in ["--", "Z0", "0000", "@@@@"]:
                return Move.null()
            elif "," in san:
                raise InvalidMoveError(f"unsupported multi-leg move: {san!r}")
            else:
                raise InvalidMoveError(f"invalid san: {san!r}")

        to_SquareBoard = SquareBoardName.index(match.group(4))
        to_mask = BB_SquareBoardS[to_SquareBoard] & ~self.occupied_co[self.turn]


        p = match.group(5)
        toQueen = pieceSymbolS.index(p[-1].lower()) if p else None


        maskIn = BB_ALL
        if match.group(2):
            fileIn = FileName.index(match.group(2))
            maskIn &= bitBoardFiles[fileIn]
        if match.group(3):
            from_rank = int(match.group(3)) - 1
            maskIn &= bitBoardRanks[from_rank]


        if match.group(1):
            Piecetype = pieceSymbolS.index(match.group(1).lower())
            maskIn &= self.pieces_mask(Piecetype, self.turn)
        elif match.group(2) and match.group(3):
            move = self.find_move(
                SquareBoard(fileIn, from_rank), to_SquareBoard, toQueen)
            if move.toQueen == toQueen:
                return move
            else:
                raise IllegalMoveError(
                    f"missing piece: {san!r} in {self.fen()}")
        else:
            maskIn &= self.pawns

            # Do not allow pawn captures if file is not specified.
            if not match.group(2):
                maskIn &= bitBoardFiles[SquareBoard_file(to_SquareBoard)]

        # Match legal moves.
        matched_move = None
        for move in self.generate_legal_moves(maskIn, to_mask):
            if move.toQueen != toQueen:
                continue

            if matched_move:
                raise AmbiguousMoveError(
                    f"ambiguous san: {san!r} in {self.fen()}")

            matched_move = move

        if not matched_move:
            raise IllegalMoveError(f"illegal san: {san!r} in {self.fen()}")

        return matched_move

    def push_san(self, san: str) -> Move:
        move = self.parse_san(san)
        self.push(move)
        return move



    def is_en_passant(self, move: Move) -> bool:
        return (self.ep_SquareBoard == move.to_SquareBoard and
                bool(self.pawns & BB_SquareBoardS[move.from_SquareBoard]) and
                abs(move.to_SquareBoard - move.from_SquareBoard) in [7, 9] and
                not self.occupied & BB_SquareBoardS[move.to_SquareBoard])

    def is_capture(self, move: Move) -> bool:
        touched = BB_SquareBoardS[move.from_SquareBoard] ^ BB_SquareBoardS[move.to_SquareBoard]
        return bool(touched & self.occupied_co[not self.turn]) or self.is_en_passant(move)

    def is_zeroing(self, move: Move) -> bool:
        touched = BB_SquareBoardS[move.from_SquareBoard] ^ BB_SquareBoardS[move.to_SquareBoard]
        return bool(touched & self.pawns or touched & self.occupied_co[not self.turn] or move.drop == PAWN)

    def _reduces_castling_rights(self, move: Move) -> bool:
        cr = self.clean_castling_rights()
        touched = BB_SquareBoardS[move.from_SquareBoard] ^ BB_SquareBoardS[move.to_SquareBoard]
        return bool(touched & cr or
                    cr & Rank1 and touched & self.kings & self.occupied_co[WHITE] & ~self.promoted or
                    cr & Rank8 and touched & self.kings & self.occupied_co[BLACK] & ~self.promoted)

    def is_irreversible(self, move: Move) -> bool:
        return self.is_zeroing(move) or self._reduces_castling_rights(move) or self.has_legal_en_passant()

    def is_castling(self, move: Move) -> bool:
        if self.kings & BB_SquareBoardS[move.from_SquareBoard]:
            diff = SquareBoard_file(move.from_SquareBoard) - SquareBoard_file(move.to_SquareBoard)
            return abs(diff) > 1 or bool(self.rooks & self.occupied_co[self.turn] & BB_SquareBoardS[move.to_SquareBoard])
        return False

    def is_kingside_castling(self, move: Move) -> bool:
        return self.is_castling(move) and SquareBoard_file(move.to_SquareBoard) > SquareBoard_file(move.from_SquareBoard)

    def is_queenside_castling(self, move: Move) -> bool:
        return self.is_castling(move) and SquareBoard_file(move.to_SquareBoard) < SquareBoard_file(move.from_SquareBoard)

    def clean_castling_rights(self) -> BitBoard:
        if self._stack:
            return self.castling_rights

        castling = self.castling_rights & self.rooks
        white_castling = castling & Rank1 & self.occupied_co[WHITE]
        black_castling = castling & Rank8 & self.occupied_co[BLACK]

        if not self.chess960:
            white_castling &= (BB_A1 | BB_H1)
            black_castling &= (BB_A8 | BB_H8)


            if not self.occupied_co[WHITE] & self.kings & ~self.promoted & BB_E1:
                white_castling = 0
            if not self.occupied_co[BLACK] & self.kings & ~self.promoted & BB_E8:
                black_castling = 0

            return white_castling | black_castling
        else:
            white_king_mask = self.occupied_co[WHITE] & self.kings & Rank1 & ~self.promoted
            black_king_mask = self.occupied_co[BLACK] & self.kings & Rank8 & ~self.promoted
            if not white_king_mask:
                white_castling = 0
            if not black_king_mask:
                black_castling = 0


            white_a_side = white_castling & -white_castling
            white_h_side = BB_SquareBoardS[msb(
                white_castling)] if white_castling else 0

            if white_a_side and msb(white_a_side) > msb(white_king_mask):
                white_a_side = 0
            if white_h_side and msb(white_h_side) < msb(white_king_mask):
                white_h_side = 0

            black_a_side = (black_castling & -black_castling)
            black_h_side = BB_SquareBoardS[msb(
                black_castling)] if black_castling else BB_EMPTY

            if black_a_side and msb(black_a_side) > msb(black_king_mask):
                black_a_side = 0
            if black_h_side and msb(black_h_side) < msb(black_king_mask):
                black_h_side = 0


            return black_a_side | black_h_side | white_a_side | white_h_side

    def has_castling_rights(self, color: Color) -> bool:
        backrank = Rank1 if color == WHITE else Rank8
        return bool(self.clean_castling_rights() & backrank)

    def has_kingside_castling_rights(self, color: Color) -> bool:
        backrank = Rank1 if color == WHITE else Rank8
        king_mask = self.kings & self.occupied_co[color] & backrank & ~self.promoted
        if not king_mask:
            return False

        castling_rights = self.clean_castling_rights() & backrank
        while castling_rights:
            rook = castling_rights & -castling_rights

            if rook > king_mask:
                return True

            castling_rights &= castling_rights - 1

        return False

    def has_queenside_castling_rights(self, color: Color) -> bool:
        backrank = Rank1 if color == WHITE else Rank8
        king_mask = self.kings & self.occupied_co[color] & backrank & ~self.promoted
        if not king_mask:
            return False

        castling_rights = self.clean_castling_rights() & backrank
        while castling_rights:
            rook = castling_rights & -castling_rights

            if rook < king_mask:
                return True

            castling_rights &= castling_rights - 1

        return False

    def status(self) -> Status:
        ERROR = STATUS_VALID

        if not self.occupied:
            ERROR |= STATUS_EMPTY

        if not self.occupied_co[WHITE] & self.kings:
            ERROR |= STATUS_NO_WHITE_KING
        if not self.occupied_co[BLACK] & self.kings:
            ERROR |= STATUS_NO_BLACK_KING
        if popcount(self.occupied & self.kings) > 2:
            ERROR |= STATUS_TOO_MANY_KINGS

        if popcount(self.occupied_co[WHITE]) > 16:
            ERROR |= STATUS_TOO_MANY_WHITE_PIECES
        if popcount(self.occupied_co[BLACK]) > 16:
            ERROR |= STATUS_TOO_MANY_BLACK_PIECES

        if popcount(self.occupied_co[WHITE] & self.pawns) > 8:
            ERROR |= STATUS_TOO_MANY_WHITE_PAWNS
        if popcount(self.occupied_co[BLACK] & self.pawns) > 8:
            ERROR |= STATUS_TOO_MANY_BLACK_PAWNS


        if self.pawns & BB_BACKRANKS:
            ERROR |= STATUS_PAWNS_ON_BACKRANK

        if self.castling_rights != self.clean_castling_rights():
            ERROR |= STATUS_BAD_CASTLING_RIGHTS


        valid_ep_SquareBoard = self._valid_ep_SquareBoard()
        if self.ep_SquareBoard != valid_ep_SquareBoard:
            ERROR |= STATUS_INVALID_EP_SquareBoard


        if self.was_into_check():
            ERROR |= STATUS_OPPOSITE_CHECK


        checkers = self.checkers_mask()
        our_kings = self.kings & self.occupied_co[self.turn] & ~self.promoted
        if checkers:
            if popcount(checkers) > 2:
                ERROR |= STATUS_TOO_MANY_CHECKERS

            if valid_ep_SquareBoard is not None:
                pushed_to = valid_ep_SquareBoard ^ A2
                pushed_from = valid_ep_SquareBoard ^ A4
                occupied_before = (
                    self.occupied & ~BB_SquareBoardS[pushed_to]) | BB_SquareBoardS[pushed_from]
                if popcount(checkers) > 1 or (
                        msb(checkers) != pushed_to and
                        self._attacked_for_king(our_kings, occupied_before)):
                    ERROR |= STATUS_IMPOSSIBLE_CHECK
            else:
                if popcount(checkers) > 2 or (popcount(checkers) == 2 and ray(lsb(checkers), msb(checkers)) & our_kings):
                    ERROR |= STATUS_IMPOSSIBLE_CHECK

        return ERROR

    def _valid_ep_SquareBoard(self) -> Optional[SquareBoard]:
        if not self.ep_SquareBoard:
            return None

        if self.turn == WHITE:
            ep_rank = 5
            pawn_mask = shift_down(BB_SquareBoardS[self.ep_SquareBoard])
            seventh_rank_mask = shift_up(BB_SquareBoardS[self.ep_SquareBoard])
        else:
            ep_rank = 2
            pawn_mask = shift_up(BB_SquareBoardS[self.ep_SquareBoard])
            seventh_rank_mask = shift_down(BB_SquareBoardS[self.ep_SquareBoard])


        if SquareBoard_rank(self.ep_SquareBoard) != ep_rank:
            return None
            
        if not self.pawns & self.occupied_co[not self.turn] & pawn_mask:
            return None

        if self.occupied & BB_SquareBoardS[self.ep_SquareBoard]:
            return None

        if self.occupied & seventh_rank_mask:
            return None

        return self.ep_SquareBoard

    def is_valid(self) -> bool:
        return self.status() == STATUS_VALID

    def _ep_skewered(self, king: SquareBoard, capturer: SquareBoard) -> bool:
        assert self.ep_SquareBoard is not None

        last_double = self.ep_SquareBoard + (-8 if self.turn == WHITE else 8)

        occupancy = (self.occupied & ~BB_SquareBoardS[last_double] &
                     ~BB_SquareBoardS[capturer] | BB_SquareBoardS[self.ep_SquareBoard])

        # Horizontal attack on the fifth or fourth rank.
        horizontal_attackers = self.occupied_co[not self.turn] & (
            self.rooks | self.queens)
        if BB_RANK_ATTACKS[king][BB_RANK_MASKS[king] & occupancy] & horizontal_attackers:
            return True

        diagonal_attackers = self.occupied_co[not self.turn] & (
            self.bishops | self.queens)
        if BB_DIAG_ATTACKS[king][BB_DIAG_MASKS[king] & occupancy] & diagonal_attackers:
            return True

        return False

    def _slider_blockers(self, king: SquareBoard) -> BitBoard:
        rooks_and_queens = self.rooks | self.queens
        bishops_and_queens = self.bishops | self.queens

        snipers = ((BB_RANK_ATTACKS[king][0] & rooks_and_queens) |
                   (FileATTACKS[king][0] & rooks_and_queens) |
                   (BB_DIAG_ATTACKS[king][0] & bishops_and_queens))

        blockers = 0

        for sniper in scan_reversed(snipers & self.occupied_co[not self.turn]):
            b = between(king, sniper) & self.occupied


            if b and BB_SquareBoardS[msb(b)] == b:
                blockers |= b

        return blockers & self.occupied_co[self.turn]

    def _is_safe(self, king: SquareBoard, blockers: BitBoard, move: Move) -> bool:
        if move.from_SquareBoard == king:
            if self.is_castling(move):
                return True
            else:
                return not self.is_attacked_by(not self.turn, move.to_SquareBoard)
        elif self.is_en_passant(move):
            return bool(self.pin_mask(self.turn, move.from_SquareBoard) & BB_SquareBoardS[move.to_SquareBoard] and
                        not self._ep_skewered(king, move.from_SquareBoard))
        else:
            return bool(not blockers & BB_SquareBoardS[move.from_SquareBoard] or
                        ray(move.from_SquareBoard, move.to_SquareBoard) & BB_SquareBoardS[king])

    def _generate_evasions(self, king: SquareBoard, checkers: BitBoard, maskIn: BitBoard = BB_ALL, to_mask: BitBoard = BB_ALL) -> Iterator[Move]:
        sliders = checkers & (self.bishops | self.rooks | self.queens)

        attacked = 0
        for checker in scan_reversed(sliders):
            attacked |= ray(king, checker) & ~BB_SquareBoardS[checker]

        if BB_SquareBoardS[king] & maskIn:
            for to_SquareBoard in scan_reversed(BB_KING_ATTACKS[king] & ~self.occupied_co[self.turn] & ~attacked & to_mask):
                yield Move(king, to_SquareBoard)

        checker = msb(checkers)
        if BB_SquareBoardS[checker] == checkers:
            target = between(king, checker) | checkers

            yield from self.generate_pseudo_legal_moves(~self.kings & maskIn, target & to_mask)

            if self.ep_SquareBoard and not BB_SquareBoardS[self.ep_SquareBoard] & target:
                last_double = self.ep_SquareBoard + \
                    (-8 if self.turn == WHITE else 8)
                if last_double == checker:
                    yield from self.generate_pseudo_legal_ep(maskIn, to_mask)

    def generate_legal_moves(self, maskIn: BitBoard = BB_ALL, to_mask: BitBoard = BB_ALL) -> Iterator[Move]:
        if self.is_variant_end():
            return

        king_mask = self.kings & self.occupied_co[self.turn]
        if king_mask:
            king = msb(king_mask)
            blockers = self._slider_blockers(king)
            checkers = self.attackers_mask(not self.turn, king)
            if checkers:
                for move in self._generate_evasions(king, checkers, maskIn, to_mask):
                    if self._is_safe(king, blockers, move):
                        yield move
            else:
                for move in self.generate_pseudo_legal_moves(maskIn, to_mask):
                    if self._is_safe(king, blockers, move):
                        yield move
        else:
            yield from self.generate_pseudo_legal_moves(maskIn, to_mask)

    def generate_legal_ep(self, maskIn: BitBoard = BB_ALL, to_mask: BitBoard = BB_ALL) -> Iterator[Move]:
        if self.is_variant_end():
            return

        for move in self.generate_pseudo_legal_ep(maskIn, to_mask):
            if not self.is_into_check(move):
                yield move

    def generate_legal_captures(self, maskIn: BitBoard = BB_ALL, to_mask: BitBoard = BB_ALL) -> Iterator[Move]:
        return itertools.chain(
            self.generate_legal_moves(
                maskIn, to_mask & self.occupied_co[not self.turn]),
            self.generate_legal_ep(maskIn, to_mask))

    def _attacked_for_king(self, path: BitBoard, occupied: BitBoard) -> bool:
        return any(self._attackers_mask(not self.turn, sq, occupied) for sq in scan_reversed(path))

    def generate_castling_moves(self, maskIn: BitBoard = BB_ALL, to_mask: BitBoard = BB_ALL) -> Iterator[Move]:
        if self.is_variant_end():
            return

        backrank = Rank1 if self.turn == WHITE else Rank8
        king = self.occupied_co[self.turn] & self.kings & ~self.promoted & backrank & maskIn
        king &= -king
        if not king:
            return

        bb_c = FileC & backrank
        bb_d = FileD & backrank
        bb_f = FileF & backrank
        bb_g = FileG & backrank

        for candidate in scan_reversed(self.clean_castling_rights() & backrank & to_mask):
            rook = BB_SquareBoardS[candidate]

            a_side = rook < king
            king_to = bb_c if a_side else bb_g
            rook_to = bb_d if a_side else bb_f

            king_path = between(msb(king), msb(king_to))
            rook_path = between(candidate, msb(rook_to))

            if not ((self.occupied ^ king ^ rook) & (king_path | rook_path | king_to | rook_to) or
                    self._attacked_for_king(king_path | king, self.occupied ^ king) or
                    self._attacked_for_king(king_to, self.occupied ^ king ^ rook ^ rook_to)):
                yield self._from_chess960(self.chess960, msb(king), candidate)

    def _from_chess960(self, chess960: bool, from_SquareBoard: SquareBoard, to_SquareBoard: SquareBoard, toQueen: Optional[PieceType] = None, drop: Optional[PieceType] = None) -> Move:
        if not chess960 and toQueen is None and drop is None:
            if from_SquareBoard == E1 and self.kings & BB_E1:
                if to_SquareBoard == H1:
                    return Move(E1, G1)
                elif to_SquareBoard == A1:
                    return Move(E1, C1)
            elif from_SquareBoard == E8 and self.kings & BB_E8:
                if to_SquareBoard == H8:
                    return Move(E8, G8)
                elif to_SquareBoard == A8:
                    return Move(E8, C8)

        return Move(from_SquareBoard, to_SquareBoard, toQueen, drop)

    def _to_chess960(self, move: Move) -> Move:
        if move.from_SquareBoard == E1 and self.kings & BB_E1:
            if move.to_SquareBoard == G1 and not self.rooks & BB_G1:
                return Move(E1, H1)
            elif move.to_SquareBoard == C1 and not self.rooks & BB_C1:
                return Move(E1, A1)
        elif move.from_SquareBoard == E8 and self.kings & BB_E8:
            if move.to_SquareBoard == G8 and not self.rooks & BB_G8:
                return Move(E8, H8)
            elif move.to_SquareBoard == C8 and not self.rooks & BB_C8:
                return Move(E8, A8)

        return move

    def _transposition_key(self) -> Hashable:
        return (self.pawns, self.knights, self.bishops, self.rooks,
                self.queens, self.kings,
                self.occupied_co[WHITE], self.occupied_co[BLACK],
                self.turn, self.clean_castling_rights(),
                self.ep_SquareBoard if self.has_legal_en_passant() else None)

    def __repr__(self) -> str:
        if not self.chess960:
            return f"{type(self).__name__}({self.fen()!r})"
        else:
            return f"{type(self).__name__}({self.fen()!r}, chess960=True)"



    def __eq__(self, board: object) -> bool:
        if isinstance(board, Board):
            return (
                self.halfmove_clock == board.halfmove_clock and
                self.fmoveNum == board.fmoveNum and
                type(self).uci_variant == type(board).uci_variant and
                self._transposition_key() == board._transposition_key())
        else:
            return NotImplemented

    def apply_transform(self, f: Callable[[BitBoard], BitBoard]) -> None:
        super().apply_transform(f)
        self.clear_stack()
        self.ep_SquareBoard = None if self.ep_SquareBoard is None else msb(
            f(BB_SquareBoardS[self.ep_SquareBoard]))
        self.castling_rights = f(self.castling_rights)

    def transform(self: BoardT, f: Callable[[BitBoard], BitBoard]) -> BoardT:
        board = self.copy(stack=False)
        board.apply_transform(f)
        return board

    def apply_mirror(self: BoardT) -> None:
        super().apply_mirror()
        self.turn = not self.turn

    def mirror(self: BoardT) -> BoardT:
        board = self.copy()
        board.apply_mirror()
        return board

    def copy(self: BoardT, *, stack: Union[bool, int] = True) -> BoardT:
        board = super().copy()

        board.chess960 = self.chess960

        board.ep_SquareBoard = self.ep_SquareBoard
        board.castling_rights = self.castling_rights
        board.turn = self.turn
        board.fmoveNum = self.fmoveNum
        board.halfmove_clock = self.halfmove_clock

        if stack:
            stack = len(self.move_stack) if stack is True else stack
            board.move_stack = [copy.copy(move)
                                for move in self.move_stack[-stack:]]
            board._stack = self._stack[-stack:]

        return board

    @classmethod
    def empty(cls: Type[BoardT], *, chess960: bool = False) -> BoardT:
        """Creates a new empty board. Also see :func:`~chess.Board.clear()`."""
        return cls(None, chess960=chess960)




class PseudoLegalMoveGenerator:

    def __init__(self, board: Board) -> None:
        self.board = board

    def __bool__(self) -> bool:
        return any(self.board.generate_pseudo_legal_moves())

    def count(self) -> int:
        # List conversion is faster than iterating.
        return len(list(self))

    def __iter__(self) -> Iterator[Move]:
        return self.board.generate_pseudo_legal_moves()

    def __contains__(self, move: Move) -> bool:
        return self.board.is_pseudo_legal(move)

    def __repr__(self) -> str:
        builder = []

        for move in self:
            if self.board.is_legal(move):
                builder.append(self.board.san(move))
            else:
                builder.append(self.board.uci(move))

        sans = ", ".join(builder)
        return f"<PseudoLegalMoveGenerator at {id(self):#x} ({sans})>"


class LegalMoveGenerator:

    def __init__(self, board: Board) -> None:
        self.board = board

    def __bool__(self) -> bool:
        return any(self.board.generate_legal_moves())

    def count(self) -> int:
        return len(list(self))

    def __iter__(self) -> Iterator[Move]:
        return self.board.generate_legal_moves()

    def __contains__(self, move: Move) -> bool:
        return self.board.is_legal(move)

    def __repr__(self) -> str:
        sans = ", ".join(self.board.san(move) for move in self)
        return f"<LegalMoveGenerator at {id(self):#x} ({sans})>"


IntoSquareBoardSet = Union[SupportsInt, Iterable[SquareBoard]]


