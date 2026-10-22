/* Exercise the real explorer's events against a small DOM facade.
   Browser layout and image decoding are checked separately. No npm packages. */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const docs = path.join(__dirname, '..', 'docs');

function startExplorer() {
  const nodes = new Map();
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, {
      innerHTML: '', textContent: '', hidden: false, dataset: {},
      classList: {toggle() {}, add() {}, remove() {}},
      attributes: {}, focusCount: 0,
      setAttribute(name, value) {this.attributes[name] = value;},
      focus() {this.focusCount++;},
      querySelectorAll() {return [];},
      querySelector: selector => node(selector),
      handlers: {},
      addEventListener(type, fn) {this.handlers[type] = fn;},
      dispatch(type, value) {this.handlers[type]({target: {value}, preventDefault() {}});}
    });
    return nodes.get(id);
  };
  const filters = ['all', 'advantage', 'caution'].map(value => {
    const result = node('filter-' + value);
    result.dataset.filter = value;
    return result;
  });
  const context = {window: {}, console, document: {
    getElementById: node,
    querySelector: selector => node(selector.replace(/^#/, '')),
    querySelectorAll: selector => selector === '[data-filter]' ? filters : [],
    body: node('body')
  }};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(docs, 'assets/js/archive-data.js'), 'utf8'), context);
  vm.runInContext(fs.readFileSync(path.join(docs, 'assets/js/explorer.js'), 'utf8'), context);
  const ids = () => [...node('photoGrid').innerHTML.matchAll(/data-dish="([^"]+)"/g)].map(m => m[1]);
  return {node, ids, rows: context.window.DEMO_DATA.rows, models: context.window.DEMO_DATA.models};
}

test('expanded gallery follows filters and resets pagination', () => {
  const {node, ids, rows, models} = startExplorer();
  node('photoGalleryToggle').dispatch('click');
  assert.equal(node('galleryPhotoCount').textContent, '48 / 507');
  node('galleryLoadMore').dispatch('click');
  assert.equal(ids().length, 96);
  node('filter-caution').dispatch('click');
  const grounded = models.findIndex(m => m.code === 'grounded');
  const expected = rows.filter(r => r.n[grounded].filter((v, i) => v !== null && Math.abs(v-r.f[i]) <= 1).length >= 4 && Math.abs(r.f[0]-r.gt[0]) >= 30);
  assert.equal(expected.length, 236);
  assert.equal(node('galleryPhotoCount').textContent, '48 / 236');
  assert.equal(ids().length, 48);
  assert.ok(ids().every(id => expected.some(r => r.id === id)));
  node('filter-all').dispatch('click');
  assert.equal(node('galleryPhotoCount').textContent, '48 / 507');
});

test('expanded gallery follows search, empty results and ordering', () => {
  const {node, ids, rows} = startExplorer();
  node('photoGalleryToggle').dispatch('click');
  node('search').dispatch('input', 'dish_1566416701');
  assert.deepEqual(ids(), ['dish_1566416701']);
  assert.equal(node('galleryPhotoCount').textContent, '1 / 1');
  assert.equal(node('galleryLoadMore').hidden, true);
  node('search').dispatch('input', 'no-matching-dish-fixture');
  assert.equal(ids().length, 0);
  assert.equal(node('galleryPhotoCount').textContent, '0 / 0');
  node('search').dispatch('input', '');
  node('sort').dispatch('change', 'id');
  assert.deepEqual(ids(), Array.from(rows, r => r.id).sort().slice(0, 48));
});

test('model changes restore focus to the changed selector for all six models', () => {
  const {node, rows, models} = startExplorer();
  const selected = rows.find(r => r.id === 'dish_1566416701');
  for (const selector of ['#modelA', '#modelB']) {
    models.forEach((model, index) => {
      const other = selector === '#modelA' ? '#modelB' : '#modelA';
      const otherFocus = node(other).focusCount;
      node(selector).dispatch('change', model.code);
      assert.equal(node(selector).focusCount, index + 1);
      assert.equal(node(other).focusCount, otherFocus);
      assert.ok(node('detail').innerHTML.includes(selected.c[index]));
    });
  }
});

test('dish choices retain button semantics and filters expose one active state', () => {
  const {node} = startExplorer();
  assert.equal((node('dishList').innerHTML.match(/<div role="listitem"><button /g) || []).length, 24);
  assert.doesNotMatch(node('dishList').innerHTML, /<button[^>]*role="listitem"/);
  for (const active of ['advantage', 'caution', 'all']) {
    node('filter-' + active).dispatch('click');
    for (const filter of ['all', 'advantage', 'caution']) {
      assert.equal(node('filter-' + filter).attributes['aria-pressed'], String(filter === active));
    }
  }
});
