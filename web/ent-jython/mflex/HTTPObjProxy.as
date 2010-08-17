package mflex
{
import flash.events.Event;
import flash.events.EventDispatcher;
import flash.events.IEventDispatcher;
import flash.utils.Proxy;
import flash.utils.flash_proxy;
import mx.events.PropertyChangeEvent;
import mx.events.PropertyChangeEventKind;
use namespace flash_proxy;
   
[Bindable("propertyChange")]
dynamic public class HTTPObjProxy extends Proxy implements IEventDispatcher
{
        protected var url:String
        protected var vars:Object;
        protected var eventDispatcher:EventDispatcher;
       
        public function HTTPObjProxy(url:String) {
                this.url = url;
                httpRpc = {};
                eventDispatcher = new EventDispatcher(this);
        }
       
        flash_proxy override function getProperty(name:*):* {
                if(httpRpc.hasOwnProperty(name))
                        return httpRpc[name].result;
                else
                        return null;
        }
       
        flash_proxy override function callProperty(name:*, ...rest):* {
                var rpc:HTTPRpc;
                if(httpRpc.hasOwnProperty(name)) {
                        rpc = httpRpc[name];
                }else {
                        rpc = HTTPRpc(url + '/' + name);
                        httpRpc[name] = rpc;
                        BindingUtils.bindSetter(function(result:Object):void{valueChangeNotify(name, result);}, rpc, "result");
                }
                rpc.call(rest[0]);
        }

        public function valueChangeNotify(name:String, value:*):void {
                var kind:String = PropertyChangeEventKind.UPDATE;
                dispatchEvent(new PropertyChangeEvent(PropertyChangeEvent.PROPERTY_CHANGE, false, false, kind, name, null, value, this));
        }
        
        public function hasEventListener(type:String):Boolean {
                return eventDispatcher.hasEventListener(type);
        }
       
        public function willTrigger(type:String):Boolean {
                return eventDispatcher.willTrigger(type);
        }
       
        public function addEventListener(type:String, listener:Function, useCapture:Boolean=false, priority:int=0.0, useWeakReference:Boolean=false):void {
                eventDispatcher.addEventListener(type, listener, useCapture, priority, useWeakReference);
        }
       
        public function removeEventListener(type:String, listener:Function, useCapture:Boolean=false):void {
                eventDispatcher.removeEventListener(type, listener, useCapture);
        }
       
        public function dispatchEvent(event:Event):Boolean {
                return eventDispatcher.dispatchEvent(event);
        }
}
}
