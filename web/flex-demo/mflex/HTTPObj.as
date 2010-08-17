package mflex
{
import flash.utils.Proxy;
import mx.rpc.http.HTTPService; 
import mx.rpc.events.ResultEvent; 
import mx.rpc.events.FaultEvent;
import com.adobe.serialization.json.JSON;
import mflex.Log;
import mflex.HTTPRpc;

public dynamic class HTTPObj {
        public var url:String;
                
        private var service:HTTPService;
        public function HTTPObj(url:String, methods:Array) {
                this.url = url;
                for each(var method:String in methods){
                                this[method] = HTTPRpc(url + '/' + method);
                        }
        }
}
}