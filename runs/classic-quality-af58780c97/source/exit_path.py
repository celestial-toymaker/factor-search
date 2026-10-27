"""退出模块专用的后续行情访问：以真实撮合规则重放单证券清仓路径。"""
from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR

import numpy as np

from .events import EventSource, at
from .ledger import Ledger, Position, ZERO
from .rules import legal_quantity, split_quantity
from .utils import dec


class ActualExitPath:
    def __init__(self, broker, known_events=lambda: ()):
        self.broker = broker
        self.data = broker.data
        self.config = broker.config
        self.known_events = known_events
        self.last = max(i for i, day in enumerate(self.data.dates) if day <= self.config.end)
        self.cache = {}

    def events(self, security):
        # 公告仍按当前可获知时间截断；后续行情访问仅用于本模块。
        relevant = {security}
        events = tuple(self.known_events())
        for event in events:
            if event.security in relevant and event.payload.get('target_security'):
                relevant.add(event.payload['target_security'])
        return tuple(event for event in events
                     if event.event_id not in self.broker.ledger.processed_events
                     and event.type not in {'user', 'order_cancel', 'order_replace'}
                     and (event.security in relevant or event.security is None))

    def path(self, security):
        column = self.data.index[security]
        key = (security, repr(self.broker.ledger.rules), repr(self.broker.ledger.status_overrides.get(security, {})),
               self.config.participation, self.config.accounting_mode)
        if self.data.arrays['amt'].flags.writeable:
            key += tuple(self.data.arrays[name][:self.last + 1, column].tobytes()
                         for name in ('open', 'close', 'amt', 'cumadj'))
            key += (self.broker.status.open_values[:self.last + 1, column].tobytes(),)
        if key in self.cache:
            return self.cache[key]
        result = []
        previous_factor = Decimal(1)
        dummy = type('ExitSell', (), {'side': 'sell', 'security': security})()
        for row in range(self.last + 1):
            context = self.broker.opening_context(row, security)
            factor = self.data.arrays['cumadj'][row, column]
            if np.isfinite(factor) and factor > 0:
                previous_factor = dec(factor)
            capacity = self.broker.capacity(row - 1, column) if row else ZERO
            reason = self.broker.block_reason(row, dummy, context)
            price = context[1]
            raw = int(capacity / price) if price and not reason else 0
            result.append({'row': row, 'date': self.data.dates[row], 'price': price,
                           'factor': previous_factor, 'capacity': capacity, 'raw_shares': raw, 'reason': reason})
        self.cache[key] = result
        return result

    def account(self, security, position, row):
        original = self.broker.ledger
        ledger = Ledger(str(original.available), self.config.accounting_mode, self.data.dates[row])
        ledger.positions[security] = deepcopy(position)
        ledger.positions[security].reserved = ZERO
        ledger.balances[f'asset:cost:{security}'] = position.cost
        ledger.balances['equity:exit_projection'] = -position.cost
        for name in ('status_overrides', 'rules', 'entitlements', 'processed_events'):
            setattr(ledger, name, deepcopy(getattr(original, name)))
        ledger.receivables = {k: deepcopy(v) for k, v in original.receivables.items() if v.get('security') == security}
        ledger.pending = {k: deepcopy(v) for k, v in original.pending.items() if v.get('security') == security}
        ledger.snapshots = {day: {security: values[security]} for day, values in original.snapshots.items() if security in values}
        return ledger

    def replay(self, security, position, row, start, events):
        from .execution import Broker
        ledger = self.account(security, position, row)
        broker = Broker(self.config, self.data, self.broker.features, self.broker.status, ledger)
        source = EventSource(events)

        def due(day, phase):
            ledger.day, ledger.phase = day, phase
            for event in source.due(at(day, phase)):
                ledger.apply(event)
                broker.on_action(event)

        def clear():
            return (not ledger.positions and not any(v['amount'] for v in ledger.receivables.values())
                    and not any(v['quantity'] for v in ledger.pending.values()))

        first = row + 1 if events else start
        if not events and self.config.accounting_mode == 'total_return':
            for index in range(row + 1, start):
                factor = self.data.arrays['cumadj'][index, self.data.index[security]]
                if np.isfinite(factor) and factor > 0:
                    ledger.apply_factor(security, factor)
        for index in range(first, self.last + 1):
            previous = index - 1
            day = self.data.dates[index]
            ledger.day, ledger.phase = self.data.dates[previous], 'decision'
            if index >= start:
                for symbol, holding in list(ledger.positions.items()):
                    if symbol in self.data.index and holding.sellable(day) > 0:
                        broker.submit(symbol, 'sell', min(holding.quantity, holding.sellable(day)), previous, index,
                                      '实际可卖路径退出重放')
            ledger.unlock(day)
            due(day, 'pre_open')
            if self.config.accounting_mode == 'total_return':
                for symbol in list(ledger.positions):
                    column = self.data.index.get(symbol)
                    if column is None:
                        continue
                    factor = self.data.arrays['cumadj'][index, column]
                    if np.isfinite(factor) and factor > 0:
                        old = ledger.positions[symbol].factor
                        ledger.apply_factor(symbol, factor)
                        broker.adjust_factor(symbol, dec(factor) / old)
            due(day, 'open')
            ledger.unlock(day)
            broker.match_open(index)
            quotes = {symbol: self.data.arrays['close'][index, self.data.index[symbol]]
                      for symbol in ledger.positions if symbol in self.data.index
                      and np.isfinite(self.data.arrays['close'][index, self.data.index[symbol]])}
            due(day, 'close')
            ledger.mark(quotes, day)
            for phase in ('rights', 'decision', 'settlement'):
                due(day, phase)
            if clear() and not events:
                break
        return {'feasible': clear(),
                'terminal_quantity': str(sum((p.quantity for p in ledger.positions.values()), ZERO)),
                'first_fill': min((trade['date'] for trade in broker.trades), default=None),
                'last_fill': max((trade['date'] for trade in broker.trades), default=None),
                'fills': [{'date': t['date'], 'quantity': t['quantity'], 'price': t['price'],
                           'capacity': t['capacity'], 'notional': t['notional']} for t in broker.trades]}

    def plan(self, security, position, row):
        future = list(range(row + 1, self.last + 1))
        if position.quantity <= 0:
            return {'feasible': True, 'latest_start': None, 'required_sessions': 0, 'terminal_quantity': '0', 'fills': []}
        if not future:
            return {'feasible': False, 'latest_start': None, 'required_sessions': None,
                    'terminal_quantity': str(position.quantity), 'fills': []}
        events = self.events(security)
        candidate = self.last
        if not events:
            path = self.path(security)
            normalized = max(ZERO, position.quantity - 1) / (position.factor if self.config.accounting_mode == 'total_return' else 1)
            available = ZERO
            candidate = None
            for index in reversed(future):
                cell = path[index]
                unit = cell['factor'] if self.config.accounting_mode == 'total_return' else Decimal(1)
                available += dec(cell['raw_shares']) / unit
                if not cell['reason'] and (position.quantity < 1 or available >= normalized):
                    candidate = index
                    break
            if candidate is None:
                return {'feasible': False, 'latest_start': None, 'required_sessions': None,
                        'terminal_quantity': str(position.quantity), 'fills': []}
        for start in range(candidate, row, -1):
            replay = self.replay(security, position, row, start, events)
            if replay['feasible']:
                finish = self.data.date_index.get(replay['last_fill'], start)
                return {**replay, 'latest_start': self.data.dates[start], 'required_sessions': finish - start + 1}
        return {**replay, 'latest_start': None, 'required_sessions': None}

    def prospective(self, security, requested_total, row):
        next_row = row + 1
        if next_row >= self.last:
            return False
        original = self.broker.ledger.positions.get(security)
        held = original.quantity if original else ZERO
        requested = max(ZERO, dec(requested_total) - held)
        context = self.broker.opening_context(next_row, security)
        dummy = type('ExitBuy', (), {'side': 'buy', 'security': security})()
        if self.broker.block_reason(next_row, dummy, context):
            return False
        price, factor = context[1], context[3]
        current_factor = self.data.arrays['cumadj'][row, self.data.index[security]]
        ratio = factor / dec(current_factor) if self.config.accounting_mode == 'total_return' else Decimal(1)
        rule = self.broker.rule(security)
        pieces = split_quantity(security, requested, 'buy', rule)
        maximum = sum(legal_quantity(security, dec(piece) * ratio, 'buy', rule=rule)
                      if abs(ratio - 1) >= Decimal('0.0001') else piece for piece in pieces)
        raw = min(maximum, int(self.broker.capacity(row, self.data.index[security]) / price))
        incoming = raw // rule.step * rule.step
        if incoming < rule.minimum:
            return False
        position = deepcopy(original) if original else Position(security)
        if self.config.accounting_mode == 'total_return':
            position.quantity *= ratio
            position.lots = [(day, quantity * ratio) for day, quantity in position.lots]
        position.quantity += incoming
        position.factor = factor
        position.reserved = ZERO
        position.lots.append((self.data.dates[next_row + 1], dec(incoming)))
        return self.plan(security, position, next_row)['feasible']
