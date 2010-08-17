package hadoop {
import mx.binding.utils.*;
import mflex.*;
import flash.net.*;
import flash.events.*;
        
public class FileList {
[Bindable]
        public var fileList:Array = [];
        private var getRpc:HTTPRpc;
        private var file:FileReference;
        private var getUrl:String = "/fileExplorer/get";
        private var uploadUrl:String = "/fileExplorer/add";
        public function FileList() {
                getRpc = new HTTPRpc(getUrl);
                BindingUtils.bindProperty(this, "fileList", getRpc, "result");
                        
                file = new FileReference();
                file.addEventListener(Event.SELECT, file_select);
                file.addEventListener(ProgressEvent.PROGRESS, file_progress);
                file.addEventListener(Event.COMPLETE, file_complete);

                getRpc.call();
        }
        public function get():void {
                getRpc.call();
        }

        public function upload():void {
                file.browse();
        }

        private function file_select(evt:Event):void {
                try {
                        file.upload(new URLRequest(uploadUrl), "fileContent");
                } catch (err:Error) {
                        Log.trace("ERROR: unable to upload file.");
                }
        }

        private function file_progress(evt:ProgressEvent):void {
                Log.trace("in progress.\n");
        }

        private function file_complete(evt:Event):void {
                Log.trace("complete.\n");
        }
}
}