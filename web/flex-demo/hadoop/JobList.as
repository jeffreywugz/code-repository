package hadoop {
import flash.utils.*;
import flash.net.*;
import flash.events.*;
import mx.binding.utils.*;
import mx.collections.ArrayCollection;
import com.adobe.serialization.json.JSON;
import mflex.*;

public class JobList {
[Bindable]
        public var jobList:ArrayCollection = new ArrayCollection();
[Bindable]
        public var uploadState:String = "NA";
[Bindable]
        public var jarName:String = "Invalid";
        public var getUrl:String = "/jobList/get";
        public var saveUrl:String = "/jobList/save";
        public var submitUrl:String = "/jobList/submit";
        public var uploadUrl:String = "/jobList/upload";
        public var getRpc:HTTPRpc;
        public var saveRpc:HTTPRpc;
        public var submitRpc:HTTPRpc;
        private var file:FileReference;
                
        public function JobList() {
                getRpc = new HTTPRpc(getUrl);
                saveRpc = new HTTPRpc(saveUrl);
                submitRpc = new HTTPRpc(submitUrl);
                // BindingUtils.bindSetter(function(list:Array):void{ jobList = new ArrayCollection(list);}, getRpc, "result");
                // get();
                BindingUtils.bindSetter(updateJobStatus, submitRpc, "result");
                file = new FileReference();
                file.addEventListener(Event.SELECT, file_select);
                file.addEventListener(ProgressEvent.PROGRESS, file_progress);
                file.addEventListener(Event.COMPLETE, file_complete);
        }
        public function get():void {
                getRpc.call();
        }
        public function save():void {
                saveRpc.call({value:jobList.source});
        }
        public function submit(cluster:String, jar:String, className:String, input:String, output:String, args:String):void {
                var job:Object = {cluster:cluster, jar:jar, className:className, input:input, output:output, args:args};
                job['mpLabel']='Executing...';
                job['mpTotal']=0;
                jobList.addItem(job);
                submitRpc.call({job:job});
        }
        public function del(index:int):void {
                jobList.removeItemAt(index);
        }
        public function report(index:int):void {
                var url:String = "/jobList/report";
                var request:URLRequest = new URLRequest(url);
                var variables:URLVariables = new URLVariables();
                variables.job = JSON.encode(jobList.getItemAt(index));
                request.data = variables;
                navigateToURL(request);
        }
        public function jobID(job:Object):String {
                return job.cluster + job.jar + job.className + job.input + job.output + job.args;
        }
        public function updateJobStatus(job:Object):void {
                if(!(job && job.hasOwnProperty('mpLabel')))return;
                for (var i:String in jobList){
                        var info:Object;
                        if(jobID(job)!=jobID(jobList[i]))continue;
                        jobList[i]['mpLabel']= job.mpLabel;
                        jobList[i]['mpTotal']= job.mpTotal;
                        jobList.itemUpdated(jobList[i]);
                        return;
                }
                return;
        }
        
        public function uploadJar():void {
                file.browse();
        }
        private function file_select(evt:Event):void {
                try {
                        file.upload(new URLRequest(uploadUrl), "fileContent");
                        jarName = file.name;
                } catch (err:Error) {
                        Log.trace("ERROR: unable to upload file.");
                }
        }

        private function file_progress(evt:ProgressEvent):void {
                uploadState = "in progress...";
        }

        private function file_complete(evt:Event):void {
                uploadState = "upload complete.";
        }
}
}