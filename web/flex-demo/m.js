function get_query_args(){
    var query = location.search.substring(1);
    return parse_query_string(query);
}

function parse_query_string(query){
    var args = {};
    var pairs = query.split("&");
    for(var i in pairs){
        var pos = pairs[i].indexOf("=");
        var argname = pairs[i].substring(0,pos);
        var value = pairs[i].substring(pos+1);
        args[argname] = decodeURIComponent(value);
    }
    return args;
}

function repr(obj) {
    return JSON.stringify(obj);
}

function error_format(msg, url, lineno){
    return repr([msg, url, lineno]);
}

function onerror(msg, url, lineno) {
    alert(error_format(msg, url, lineno));
    return true;
};

window.onerror = onerror;