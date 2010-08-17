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
        
        public static function trace(format:String, ... args):void {
                log += StringUtil.substitute(format, args);
        }
                
        public static function traceCall(func:String, args:Array):void {
                var argsRepr:String = JSON.encode(args);
                Log.trace("{0}({1})\n", func, argsRepr.substring(1, argsRepr.length-1));
        }

        public static function traceDebug(id:String, debug:String):void {
                Log.trace("{0}: {1}\n", id, debug);
        }
}
}
