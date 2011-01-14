//////////////////////////////////////// Common ////////////////////////////////////////
function product2(A, B) Array.concat.apply([], [[x.concat([y]) for each(y in B)] for each(x in A)])
function product(seq) seq.reduce(product2, [[]])
function rproduct(seq) product(seq.reverse()).map(function(i) i.reverse())
function dict(pairs){ var result = {}; for each([k,v] in pairs) result[k] = v;  return result;}
function dictSlice(d, keys) dict([[x, d[x]] for each(x in keys)])
function min(seq, comp) seq.length > 0 && seq.sort(comp)[0]
function sum(seq) seq.reduce(function(v1, v2) v1+v2)
function repr(obj) JSON.stringify(obj)
function bind(obj, attrs) {for ([k,v] in Iterator(attrs))obj[k]=v; return obj;}
function watch(obj, keys, f){
    for each(let x in keys)
        obj.watch(x, function(i,o,n){ let attrs = dictSlice(obj, keys); attrs[i] = n; f(attrs); return n});
}
String.prototype.format = function(dict) this.replace(/\${(\w+)}/g, function(m,k) repr(dict[k]));
Number.prototype.__iterator__ = function(){ for ( let i = 0; i < this; i++ )yield i;};

function log() window.console &&  console.log.apply(null, arguments)
function $(id) document.getElementById(id)
function isHotKey(e, key) String.fromCharCode(e.charCode).toUpperCase() == key.toUpperCase()
function bindKey(w, key, handler) w.addEventListener('keypress', function(e) isHotKey(e, key) && handler(e), false);

//////////////////////////////////////// Transform ////////////////////////////////////////
function transpose(A) [[row[i] for each([_,row] in Iterator(A))]  for(i in A[0].length)]
function innerProduct(A, B) sum([A[i]*B[i] for(i in A.length)])
function transform(v, B) transpose(B).map(function(col) innerProduct(v, col))
function matMul(A, B) A.map(function(row) transform(row, B))
function xyzk2xy([x, y, z, k]) [x/k, y/k]

var [sin, cos] = [Math.sin, Math.cos];
function translate([dx, dy, dz]) [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [dx, dy, dz, 1]]
function scale([sx, sy, sz]) [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]]
function rotateX(a) [[1, 0, 0, 0], [0, cos(a), sin(a), 0], [0, -sin(a), cos(a), 0], [0, 0, 0, 1]]
function rotateY(b) [[cos(b), 0, -sin(b), 0], [0, 1, 0, 0], [sin(b), 0, cos(b), 0], [0, 0, 0, 1]]
function rotateZ(c) [[cos(c), sin(c), 0, 0], [-sin(c), cos(c), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
function rotate([a, b, c]) [rotateX(a), rotateY(b), rotateZ(c)].reduce(matMul)
function projection(D) [[1,0,0,0], [0,1,0,0], [0,0,1,1.0/D], [0,0,0,1]]
function place(L, xyz) matMul(scale([L, L, L]), translate(xyz))

//////////////////////////////////////// Polygon ////////////////////////////////////////
function polygon(ctx, _ps){
    var ps = _ps.map(xyzk2xy);
    ctx.save();
    bind(ctx, {strokeStyle:'black', lineWidth: '3', lineCap: 'round', lineJoin: 'round'});
    ctx.fillStyle = _ps.color;
    ctx.beginPath();
    var [x0, y0] = ps[0];
    ctx.moveTo(x0, y0); 
    for each(let [x, y] in ps.slice(1))ctx.lineTo(x, y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
}

function polygons(ctx, planes){
    planes.map(function(p) polygon(ctx, p));
}

//////////////////////////////////////// RubikRenderer ////////////////////////////////////////
function cubePlane([px, py, pz]){
    function bin2gray(i) (i^(i<<1)) >> 1
    var vertexes = [let(j=bin2gray(i))[(j&1)*2-1, (j&2)-1, ((j&4)>>1)-1,1] for(i in 8)];
    return [xyzk for each(xyzk in vertexes) if(innerProduct(xyzk, [px, py, pz, 0])==1)];
}

function rubikPlaneIds(){
    var cubes = rproduct([[-1,0,1], [-1,0,1], [-1,0,1]]);
    var planes = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];
    return product([cubes, planes]);
}

function rubikPlane([cId, pId], abc, axle, angles){
    function refPoint(cId){ let p = [0, 0, 0, 1]; p[axle] = cId[axle]; return p;}
    var axleRotate = [rotateX, rotateY, rotateZ][axle](angles[cId[axle] + 1]);
    let plane = [cubePlane(pId), place(0.5, cId), axleRotate, rotate(abc)].reduce(matMul);
    return bind(plane, {ref: transform(refPoint(cId), rotate(abc))});
}

function rubikPlaneSorter(p1,p2){
    function maxZ(plane) min(plane.map(function([x,y,z,k])z/k), function(z1, z2) z2-z1)
    function refZ(plane) let([x,y,z,k]=plane.ref) z/k
    return (refZ(p1) == refZ(p2))? (maxZ(p1) - maxZ(p2)): (refZ(p1) - refZ(p2));
}

function rubikPlaneColoring([cId, pId]){
    return innerProduct(innerProduct(cId, pId) == 1? pId: [0,0,0], [1,2,3]) + 3;
}

function rubikColoredPlane(id, abc, axle, angles, permutation){
    return bind(rubikPlane(id, abc, axle, angles), {color:rubikPlaneColoring(permutation[id])});
}

function rubikRender(abc, axle, angles, permutation){
    return [rubikColoredPlane(id, abc, axle, angles, permutation) for each(id in rubikPlaneIds())].sort(rubikPlaneSorter);
}

function makeRubikRenderer(ctx, L, xyz, D, cm){
    cm = cm || ['white', 'red', 'blue', 'gray', 'orange', 'green', 'yellow'];
    return function render(abc, axle, angles, permutation){
        ctx.clearRect(0, 0, ctx.width, ctx.height); 
        let planes = rubikRender(abc, axle, angles, permutation);
        planes = [bind([p, place(L, xyz)].reduce(matMul), {color:cm[p.color]}) for each(p in planes)]
        polygons(ctx, planes);
    }
}

//////////////////////////////////////// Permute ////////////////////////////////////////
function permuteMul(A, B) dict([[k, B[v]] for each([k,v] in Iterator(A))])

function rubikPlaneAxleTrans(axle, LR){
    function comp(x, y) x<y? -1: x>y? 1: 0
    function xyzTrans([i,j], axle, LR) (i == axle && j == axle)? 1: (i != axle && j != axle)? comp(i,j)*LR: 0
    return [[xyzTrans([i,j], axle, LR) for(j in 3)] for(i in 3)];
}

function rubikPlanePermuted([cId, pId], axle, level, LR){
    return (cId[axle] != level)? [cId, pId]: matMul([cId, pId], rubikPlaneAxleTrans(axle, LR));
}

function rubikPermutation(axle, level, LR){
    return dict([[x, rubikPlanePermuted(x, axle, level, LR)] for each(x in rubikPlaneIds())]);
}

function rubikPermuted(permute, axle, level, LR) permuteMul(permute, rubikPermutation(axle, level, LR))
function rubikIdentityPermutation() dict([[x,x] for each(x in rubikPlaneIds())])

//////////////////////////////////////// Rubik ////////////////////////////////////////
function Rubik(render, permutation, abc, axle, angles){
    permutation = permutation || rubikIdentityPermutation();
    abc = abc || [0.2,-0.2,0.2];
    axle = axle || 0;
    angles = angles || [0, 0, 0];
    bind(this, {render: render, permutation: permutation, abc: abc , axle: axle, angles: angles});
}
Rubik.prototype.draw = function(){
    this.render(this.abc, this.axle, this.angles, this.permutation);
    this.drawHook && this.drawHook();
}

Rubik.prototype.resetView = function() this.abc = [0.2, -0.2, 0.2];
Rubik.prototype.resetPermutation = function() this.permutation = rubikIdentityPermutation();
Rubik.prototype.viewRotate = function(axle, delta) this.abc[axle] += delta;
Rubik.prototype.setAxle = function(axle) [this.axle, this.angles] = [axle, [0, 0, 0]];
//Rubik.prototype.permute = function(level, LR) this.angles[level] += LR*0.1;
Rubik.prototype.permute = function(level, LR) this.permutation = rubikPermuted(this.permutation, this.axle, level, LR);
