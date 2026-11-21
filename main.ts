// Room-decomposed push-optimal Sokoban solver.
// Idea: a generated level is a hallway with 1-4 rooms off it; a box pushed
// out of a room can never come back, so each room is an independent problem
// and the level optimum is the sum of the rooms' optima. Each room (3-4
// boxes) is solved with A* over pushes: a state is the boxes plus the
// player's reachable region, cost is pushes, heuristic is the best
// assignment of boxes to goals by Manhattan distance (admissible: one push
// moves one box by at most 1). Dead squares (from which no push reaches a
// goal) and frozen 2x2 blocks are pruned. No clock, no randomness, no I/O
// outside /input and /output: the search budget is a fixed state count.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MAX_STATES = 250_000;
// [dx, dy, walk letter, push letter]
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

type Level = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++)
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a box alone can be pushed onto some goal.
function liveSquares(lv: Level): Uint8Array {
  const { w, wall, goals } = lv;
  const live = new Uint8Array(wall.length);
  const q = [...goals];
  for (const g of goals) live[g] = 1;
  for (let i = 0; i < q.length; i++) {
    const x = q[i]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y < 0 || y >= wall.length || y - d < 0 || y - d >= wall.length) continue;
      if (!wall[y] && !wall[y - d] && !live[y]) { live[y] = 1; q.push(y); }
    }
  }
  return live;
}

// Player's reachable region from `from`, avoiding walls and boxes.
// Returns lowest square; `seen` is stamped with `stamp` on reachable cells.
function reach(lv: Level, occ: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  const { w, wall } = lv;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (q >= 0 && q < wall.length && wall[q] === 0 && occ[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

function frozen(lv: Level, occ: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = lv;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const cells = [a, a + 1, a + w, a + w + 1];
    if (cells.some((c) => c < 0 || c >= wall.length)) continue;
    if (!cells.every((c) => wall[c] || occ[c])) continue;
    if (cells.some((c) => occ[c] && !goal[c])) return true;
  }
  return false;
}

// Shortest walk from `from` to `to` (lower-case letters), avoiding walls/boxes.
function walkTo(lv: Level, occ: Uint8Array, from: number, to: number): string {
  const { w, wall } = lv;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const q = [from];
  prev[from] = from;
  for (let i = 0; i < q.length; i++) {
    const p = q[i]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const s = p + dy * w + dx;
      if (s < 0 || s >= wall.length || wall[s] || occ[s] || prev[s] !== -1) continue;
      prev[s] = p;
      how[s] = letter;
      q.push(s);
    }
  }
  if (prev[to] === -1) return '';
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a = [...Array(n).keys()];
  const gen = (k: number) => {
    if (k === n) { out.push([...a]); return; }
    for (let i = k; i < n; i++) { [a[k], a[i]] = [a[i]!, a[k]!]; gen(k + 1); [a[k], a[i]] = [a[i]!, a[k]!]; }
  };
  gen(0);
  return out;
}

// Push-optimal solution {pushes, moves} for one level, or null if the
// state budget runs out. A* over pushes; first solved state popped is optimal.
function solvePushOptimal(lv: Level, maxStates = MAX_STATES): { pushes: number; moves: string } | null {
  const { w, wall, goals } = lv;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(lv);
  if (lv.boxes.some((b) => !live[b])) return null;
  const solved = (bs: number[]): boolean => bs.every((b) => goal[b]);
  if (solved(lv.boxes)) return { pushes: 0, moves: '' };
  const man = (a: number, b: number): number =>
    Math.abs((a % w) - (b % w)) + Math.abs(Math.floor(a / w) - Math.floor(b / w));
  const perms = permutations(goals.length);
  const heur = (bs: number[]): number => {
    let best = Infinity;
    for (const p of perms) {
      let s = 0;
      for (let i = 0; i < bs.length; i++) s += man(bs[i]!, goals[p[i]!]!);
      if (s < best) best = s;
    }
    return best;
  };
  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const keyOf = (bs: number[], low: number): number => {
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };
  const sBoxes: number[][] = [lv.boxes];
  const sPlayer: number[] = [lv.player];
  const sG: number[] = [0];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushTo: number[] = [-1];
  const pushDir: number[] = [-1];
  const best = new Map<number, number>();
  for (const b of lv.boxes) occ[b] = 1;
  const low0 = reach(lv, occ, lv.player, seen, ++stamp, stack);
  for (const b of lv.boxes) occ[b] = 0;
  best.set(keyOf(lv.boxes, low0), 0);
  // Binary heap on [f, g, state]; ties: larger g first, then earlier state.
  const heap: [number, number, number][] = [[heur(lv.boxes), 0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const hPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  let found = -1;
  while (heap.length > 0) {
    const [, g, s] = hPop();
    if (g !== sG[s]) continue;
    const bs = sBoxes[s]!;
    if (solved(bs)) { found = s; break; }
    for (const b of bs) occ[b] = 1;
    const region = ++stamp;
    reach(lv, occ, sPlayer[s]!, seen, region, stack);
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * w + dx;
        const t = b + d;
        if (t < 0 || t >= size) continue;
        if (seen[b - d] !== region || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0; occ[t] = 1;
        const fr = frozen(lv, occ, goal, t);
        let key = -1;
        let next: number[] | null = null;
        if (!fr) {
          next = bs.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(next, reach(lv, occ, b, mark, ++stamp, stack));
        }
        occ[t] = 0; occ[b] = 1;
        if (fr || next === null) continue;
        const ng = g + 1;
        const prevG = best.get(key);
        if (prevG !== undefined && prevG <= ng) continue;
        if (sBoxes.length >= maxStates) { for (const x of bs) occ[x] = 0; return null; }
        best.set(key, ng);
        sBoxes.push(next);
        sPlayer.push(b);
        sG.push(ng);
        parent.push(s);
        pushFrom.push(b);
        pushTo.push(t);
        pushDir.push(di);
        hPush([ng + heur(next), ng, sBoxes.length - 1]);
      }
    }
    for (const x of bs) occ[x] = 0;
  }
  if (found < 0) return null;
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = lv.player;
  let boxes = lv.boxes;
  for (const s of path) {
    const nextBoxes = sBoxes[s]!;
    const from = pushFrom[s]!;
    const to = pushTo[s]!;
    const d = to - from;
    for (const b of boxes) occ[b] = 1;
    moves += walkTo(lv, occ, player, from - d);
    for (const b of boxes) occ[b] = 0;
    moves += DIRS[pushDir[s]!]![3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// Split a level into rooms: cut the hallway row's dead squares (floor that
// no box can reach a goal from), each remaining component holding boxes is
// a room. Null when the level is not a set of rooms.
function roomsOf(lv: Level): { boxes: number[]; goals: number[]; done: boolean }[] | null {
  const { w, wall, goals, boxes, player } = lv;
  const live = liveSquares(lv);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: { boxes: number[]; goals: number[]; done: boolean }[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (q >= 0 && q < wall.length && !wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function solveByRooms(lv: Level): { pushes: number; moves: string } | null {
  const rooms = roomsOf(lv);
  if (rooms === null) return solvePushOptimal(lv);
  let player = lv.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = lv.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: lv.w, h: lv.h, wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const s = solvePushOptimal(sub);
    if (s === null) return null;
    pushes += s.pushes;
    moves += s.moves;
    player = sub.player;
    for (const ch of s.moves) {
      const c = ch.toLowerCase();
      player += c === 'u' ? -lv.w : c === 'd' ? lv.w : c === 'l' ? -1 : 1;
    }
    room.done = true;
  }
  return { pushes, moves };
}

function solve(text: string): string {
  const r = solveByRooms(parse(text));
  return r === null ? '' : r.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
