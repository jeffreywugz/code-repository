package mflex
{
import mx.utils.StringUtil;
import mx.utils.ObjectUtil;
import com.adobe.serialization.json.JSON;
import flash.events.Event;

public dynamic class Expr {
        public function Expr() {
        }
        
        public function eval(expr:String):Object{
                var match:Array = expr.match(/(.+?)\((.*)\)/);
                if(!match){
                        Log.traceFault("Expr.eval", "illformed input!");
                        return null;
                }
                var objRepr:String = match[1];
                var argsRepr:String = match[2];
                var objNameList:Array;
                var obj:Object;
                var args:Object;
                objNameList = objRepr.split('.');
                argsRepr = "[" + argsRepr + "]";
                Log.trace('Expr.eval: objNameList={0}, argsRepr={1}\n', JSON.encode(objNameList), argsRepr);
                obj = getObj(objNameList);
                args = JSON.decode(argsRepr);
                return obj.apply(obj, args);
        }
        public function getObj(nameList:Array):Object {
                var obj:Object = this;
                for each (var name:String in nameList){
                        if(obj.hasOwnProperty(name))
                                obj = obj[name];
                        else
                                return null;
                }
                return obj;
        }
}
}
