package hadoop {
import flash.utils.*;
import flash.events.*;
import mx.binding.utils.*;
import mx.collections.ArrayCollection;
import mflex.*;

public class Preference {
[Bindable]
        public var preference:Object;
        public var getUrl:String = "/preference/get";
        public var saveUrl:String = "/preference/save";
        public var getRpc:HTTPRpc;
        public var saveRpc:HTTPRpc;
                
        public function Preference() {
                getRpc = new HTTPRpc(getUrl);
                saveRpc = new HTTPRpc(saveUrl);
                BindingUtils.bindProperty(this, 'preference', getRpc, "result");
                getRpc.call();
        }
        public function get():void {
                getRpc.call();
        }
        public function save(vars:Object):void {
                saveRpc.call({value:vars});
        }
}
}