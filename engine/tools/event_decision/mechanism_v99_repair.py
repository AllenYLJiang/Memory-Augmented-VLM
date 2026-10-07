"""Reversible, kind-directed encoding compatibility. Never infer visual evidence."""
import copy
import re
from collections import Counter

from .mechanism_v96_contract import ROLES
from .mechanism_v98_contract import assess

VERSION = 'v99_reversible_typed_compatibility_1'


def normalize(raw):
    value = copy.deepcopy(raw)
    changes = []
    if not isinstance(value, dict): return value, changes
    entities, observations = value.get('entities'), value.get('observations')
    if not isinstance(entities, list) or not isinstance(observations, list): return value, changes
    ids = [e.get('id') for e in entities if isinstance(e, dict) and isinstance(e.get('id'), str)]
    if len(ids) != len(entities) or len(set(ids)) != len(ids): return value, changes
    def change(path, before, after, rule):
        if before != after: changes.append({'path': path, 'before': copy.deepcopy(before),
                                          'after': copy.deepcopy(after), 'rule': rule})
    def get_roles(event):
        relation = event.get('relation') if isinstance(event, dict) else None
        if not isinstance(relation, dict): return None
        if any(not isinstance(relation.get(r), list) or not all(isinstance(x, str) for x in relation[r]) for r in ROLES):
            return None
        return relation
    role_refs = [x for e in observations if get_roles(e) is not None for r in ROLES for x in e['relation'][r]]
    context_refs = [x for e in observations if isinstance(e, dict) and isinstance(e.get('context_entity_ids'), list)
                    for x in e['context_entity_ids'] if isinstance(x, str)]
    mapping = {}
    occupied = set(ids)
    def fresh(prefix):
        number = 1
        while prefix+str(number) in occupied: number += 1
        result = prefix+str(number); occupied.add(result); return result
    for uid in ids:
        if re.fullmatch(r'p0[0-9]*', uid):
            mapping[uid] = fresh('p')
        elif re.fullmatch(r'env_[A-Za-z][A-Za-z0-9_]*', uid) and uid in context_refs and uid not in role_refs:
            # env_* is a legacy opaque context handle. Renaming it does NOT assert
            # an object class, actor identity, relation argument, or new visibility.
            mapping[uid] = fresh('o')
    for i, entity in enumerate(entities):
        uid = entity['id']
        if uid in mapping:
            change(f'/entities/{i}/id', uid, mapping[uid], 'bijective_local_handle_alias')
            entity['id'] = mapping[uid]
    for i, event in enumerate(observations):
        if not isinstance(event, dict): continue
        relation = get_roles(event)
        slots = [(event, 'context_entity_ids', f'/observations/{i}/context_entity_ids')]
        if relation is not None: slots += [(relation, r, f'/observations/{i}/relation/{r}') for r in ROLES]
        for parent, field, path in slots:
            old = parent.get(field)
            if isinstance(old, list) and all(isinstance(x, str) for x in old):
                new = [mapping.get(x, x) for x in old]
                change(path, old, new, 'bijective_local_handle_alias'); parent[field] = new
    valid_ids = {e['id'] for e in assess(value)['entities'] if e['valid']}
    for i, event in enumerate(observations):
        relation = get_roles(event)
        if relation is None: continue
        flat = [x for r in ROLES for x in relation[r]]
        if len(flat) != len(set(flat)) or not all(x in valid_ids for x in flat): continue
        original_roles = {r: list(relation[r]) for r in ROLES}
        a, t, p, o, ins = (relation[r] for r in ROLES)
        kind = event.get('kind')
        rule = None
        if kind in ('person_object_interaction', 'co_presence') and not t:
            # Existing p/o prefixes and explicit kind determine this partition.
            # Never infer an actor/target pair or add/drop a necessary argument.
            can = all(x.startswith('p') for x in a) and all(x.startswith('o') for x in o+ins)
            people = a+[x for x in p if x.startswith('p')]
            objects = o+[x for x in p if x.startswith('o')]
            can = can and len(people)+len(objects)+len(ins) == len(flat)
            if kind == 'person_object_interaction': can = can and len(people) == len(objects) == 1
            else: can = can and not a and not ins and len(people)+len(objects) >= 2
            if can:
                relation.update(actor_ids=[], participant_ids=people, object_ids=objects)
                rule = 'explicit_kind_and_p_o_typed_role_partition'
        elif kind == 'object_constraint' and event.get('phase') == 'ongoing_constraint':
            if not p and not o and len(t) == 1 and len(a) <= 1 and len(ins) == 1 and ins[0].startswith('o'):
                relation.update(object_ids=list(ins), instrument_ids=[])
                rule = 'sole_constraint_object_in_explicit_ongoing_constraint'
        if rule:
            if Counter(flat) != Counter(x for r in ROLES for x in relation[r]):
                raise AssertionError('Necessary relation arguments changed')
            for r in ROLES: change(f'/observations/{i}/relation/{r}', original_roles[r], relation[r], rule)
    if undo(value, changes) != raw: raise AssertionError('Compatibility transform is not reversible')
    return value, changes


def undo(value, changes):
    result = copy.deepcopy(value)
    for item in reversed(changes):
        parts = item['path'].strip('/').split('/'); parent = result
        for key in parts[:-1]: parent = parent[int(key)] if isinstance(parent, list) else parent[key]
        key = parts[-1]
        if isinstance(parent, list): key = int(key)
        if parent[key] != item['after']: raise ValueError('Change log does not match payload')
        parent[key] = copy.deepcopy(item['before'])
    return result


def assert_evidence_unchanged(before, after, changes):
    if undo(after, changes) != before: raise AssertionError('Round trip differs')
    # Only identifiers and role-list containers may change, never descriptions,
    # textual evidence, kind, phase, presence, strength, context scope, or bins.
    for item in changes:
        if not (re.fullmatch(r'/entities/[0-9]+/id', item['path']) or
                re.fullmatch(r'/observations/[0-9]+/context_entity_ids', item['path']) or
                re.fullmatch(r'/observations/[0-9]+/relation/('+'|'.join(ROLES)+')', item['path'])):
            raise AssertionError('Non-encoding field changed')
    for left, right in zip(before['entities'], after['entities']):
        if {k: v for k, v in left.items() if k != 'id'} != {k: v for k, v in right.items() if k != 'id'}:
            raise AssertionError('Entity evidence changed')
    for left, right in zip(before['observations'], after['observations']):
        for key in set(left)-{'relation', 'context_entity_ids'}:
            if left[key] != right[key]: raise AssertionError('Observation changed: '+key)
        if isinstance(left['relation'], dict):
            for key in set(left['relation'])-set(ROLES):
                if left['relation'][key] != right['relation'][key]: raise AssertionError('Relation evidence changed')
    return True
