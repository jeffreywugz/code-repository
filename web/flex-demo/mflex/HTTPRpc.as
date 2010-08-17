package mflex
{
import flash.utils.Proxy;
import mx.rpc.http.HTTPService; 
import mx.rpc.events.ResultEvent; 
import mx.rpc.events.FaultEvent;
import com.adobe.serialization.json.JSON;
import mflex.Log;

public class HTTPRpc {
        public var url:String;
[Bindable]
        public var result:Object;
                
        private var service:HTTPService;
        public function HTTPRpc(url:String, concurrency:String="single") {
                this.url = url;
                this.service = new HTTPService();
                service.concurrency = concurrency;
                service.url = url;
                service.resultFormat = "text";
                service.addEventListener("result", onResult); 
                service.addEventListener("fault", onFault); 
        }
                
        public static function argsDumps(args:Object):Object {
                var dumpedArgs:Object = {};
                for (var p:String in args){
                        dumpedArgs[p] = JSON.encode(args[p]);
                }
                return dumpedArgs;
        }
                
        public function call(args:Object=null, method:String="POST"):void {
                service.method = method; 
                // Log.traceCall("HTTPRpc.call", arguments);
                var newArgs:Object = HTTPRpc.argsDumps(args);
                Log.traceDebug("HTTPRpc.call", service.url + "?" + JSON.encode(newArgs));
                service.send(newArgs); 
        } 
 
        public function onResult(event:ResultEvent):void {
                result = JSON.decode(event.result.toString());
        } 
 
        public function onFault(event:FaultEvent):void { 
                var fault:String = String(event.message.body).replace(/<style.*<\/style>/s, "");
                Log.traceFault("HTTPRpc", fault);
        }
}
}