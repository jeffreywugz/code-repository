package mflex
{
import mx.utils.StringUtil;
import mx.utils.ObjectUtil;
import com.adobe.serialization.json.JSON;
import flash.events.Event;
import flash.utils.*;
import mx.binding.utils.*;

public class Log {
[Bindable]
        public static var log:String = "Log Start.\n";
[Bindable]
        public static var debug:String = "Debug Start.\n";
[Bindable]
        public static var fault:String = "";
        public static var faultHook:Function = function():void{};
        public static var remoteLogs:Array = [];
        public static function addRemote(...urls):void {
                for each(var url:String in urls){
                                var rpc:HTTPRpc = new HTTPRpc(url);
                                BindingUtils.bindSetter(function(msg:String):void{ Log.trace(msg);}, rpc, "result");
                                remoteLogs.push(rpc);
                        }
        }

        public static function startTraceRemote():void {
                setInterval(updateRemoteLogs, 1000);
        }
        public static function updateRemoteLogs():void {
                for each(var rpc:HTTPRpc in remoteLogs){
                                rpc.call();
                        }
        }
        public static function trace(format:String, ... args):void {
                log += StringUtil.substitute(format, args);
        }
                
        public static function traceCall(func:String, args:Array):void {
                var argsRepr:String = JSON.encode(args);
                Log.trace("{0}({1})\n", func, argsRepr.substring(1, argsRepr.length-1));
        }

        public static function traceFault(id:String, fault:String):void {
                Log.fault = fault;
                Log.trace("{0}: Fault Info Recorded\n", id);
                Log.faultHook();
        }
        
        public static function traceDebug(id:String, debug:String):void {
                Log.debug = StringUtil.substitute("{0}: {1}\n", id, debug);
        }
}
}
