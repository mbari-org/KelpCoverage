import os
import pathlib
import sys
import time
import shutil
from typing import List, Tuple

_SENTINEL = pathlib.Path.home() / ".config" / "kelp_coverage" / ".welcomed"

def _already_seen() -> bool:
    return _SENTINEL.exists()

def _mark_seen() -> None:
    _SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    _SENTINEL.touch()

ESC          = '\x1b['
HIDE_CURSOR  = f'{ESC}?25l'
SHOW_CURSOR  = f'{ESC}?25h'
HOME         = f'{ESC}H' 
CLEAR_SCREEN = f'{ESC}2J'
RESET        = f'{ESC}0m'

def rgb(r: int, g: int, b: int) -> str:
    return f'{ESC}38;2;{r};{g};{b}m'

def bg(r: int, g: int, b: int) -> str:
    return f'{ESC}48;2;{r};{g};{b}m'

def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)

Color = Tuple[int, int, int]

def lerp_color(c1: Color, c2: Color, t: float) -> Color:
    return (lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t))

BLACK:       Color = (  0,   0,   0)
MBARI_NAVY:  Color = (  0,  48, 102)
MBARI_WHITE: Color = (232, 242, 255)

LOGO: List[str] = [
'████████████████████████████████████████████████████████████████████████',
' █████████████████████████████████████████▓▒░▒▒████████████████████████',
'  █████████████████████████████████████▓▒░  ░░▓███████████████████████',
'   ██████████████████████████████████▒░  o ░░▒███████████████████████',
'    ███████████████████████████▓▒▒▓▒░    ░░▒▓▓██████████████████████',
'     ████████████████████████▓░ ░▓░    ░░░▓▓▒██████████████████████',
'      ██████████████████████▒░░▒▒░   ░░░▒█▒▒██████████████████████',
'       ████████████████████▓░░▓▒   ░░░▒▓▓▓▓████▓█████████████████',
'        ███████████████████░ ▓░   ░░▒▓▒░ ░░░░▒▒▒▒███████████████',
'         █████████████████▒ ▒▒    ░▓▒▒▒▒░░░ ░░▒▓███████████████',
'          ███████████████▓░░▓    ░▓███▓▓▓▓▓▓██████████████████',
'           ██████████████▒ ▓░  ░▒▓███████████████████████████',
'            █████████████░▒▒   ▓█████▓▒░░▒▓█████████████████',
'             ████████████░▓░  ▒████▓░  ░▒▒▒▓███████████████',
'              ███████████░▓   ▓██▓░   ▒██▒ ▓██████████████',
'                  ███████░▒  ▒██▒░  ░▓██▒  ▓██████████',
'                   ██████▓▒ ▒█▓░   ░▓██▒  ▒██████████',
'                    ██████▒░█▒░   ░██▓░  ▒██████████',
'                     █████▒▒░   ░▒██▒  ░▓██████████',
'                      ████▓░▒  ░▓█▓░  ░▓██████████',
'                       ████▓▓▓▓██▓░ ░▓███████████',
'                        ████████▓░ ░▓▓▒░░▓██████',
'                         ███████▒░▒█▓░░▓▓▓█████',
'                          ██████░▒█▒ ▒██░▓████',
'                           █████▒▓░ ▒██░░▓███',
'                            █████▓▒▓██░░▓███',
'                             ███████▓░░▓███',
'                              █████▓░░▓███',
'                               ███▓░▒████',
'                                ██▒▓████',
'                                 ██████',
'                                  ████',
'                                   ██',
]

TITLE: List[str] = [
'███╗   ███╗██████╗  █████╗ ██████╗ ██╗',
'████╗ ████║██╔══██╗██╔══██╗██╔══██╗██║',
'██╔████╔██║██████╔╝███████║██████╔╝██║',
'██║╚██╔╝██║██╔══██╗██╔══██║██╔══██╗██║',
'██║ ╚═╝ ██║██████╔╝██║  ██║██║  ██║██║',
'╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝',
]


def play(force: bool = False) -> None:
    if not sys.stdout.isatty() or os.environ.get('NO_LAUNCH_SCREEN'):
        return
    if not force and _already_seen():
        return

    size = shutil.get_terminal_size(fallback=(80, 24))
    cols, rows = size.columns, size.lines
    
    logo_width = max(len(l) for l in LOGO)
    title_width = max(len(l) for l in TITLE)
    
    total_height = len(LOGO) + 1 + len(TITLE)
    start_row = max(1, (rows - total_height) // 2)
    logo_pad = " " * max(0, (cols - logo_width) // 2)
    title_pad = " " * max(0, (cols - title_width) // 2)

    out = sys.stdout.write

    try:
        out(HIDE_CURSOR + CLEAR_SCREEN) 

        def _build_buf(t: float) -> str:
            navy        = lerp_color(BLACK, MBARI_NAVY,  t)
            white       = lerp_color(BLACK, MBARI_WHITE, t)
            title_color = lerp_color(BLACK, MBARI_NAVY, t)

            buf = HOME + ("\n" * start_row)
            for line in LOGO:
                strip_start = len(line) - len(line.lstrip(' '))
                row = logo_pad + line[:strip_start]
                for ch in line[strip_start:]:
                    if ch == '█':
                        row += rgb(*navy) + bg(*navy) + '█'
                    elif ch in ('▓', '▒', '░'):
                        row += rgb(*navy) + bg(*white) + ch
                    else:   
                        row += bg(*white) + ' '
                buf += row + RESET + '\n'
            buf += '\n'
            for line in TITLE:
                buf += f"{title_pad}{rgb(*title_color)}{line}{RESET}\n"
            return buf

        for frame in range(25):
            out(_build_buf(frame / 24))
            sys.stdout.flush()
            time.sleep(0.04)

        time.sleep(0.8)  

        for frame in range(25):
            out(_build_buf(1.0 - frame / 24))
            sys.stdout.flush()
            time.sleep(0.03)

        _mark_seen()

    finally:
        out(CLEAR_SCREEN + HOME + SHOW_CURSOR + RESET)
        sys.stdout.flush()

if __name__ == "__main__":
    play()
