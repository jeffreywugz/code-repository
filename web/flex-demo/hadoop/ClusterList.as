package hadoop {
import flash.utils.*;
import flash.events.*;
import mx.binding.utils.*;
import mx.collections.ArrayCollection;
import mflex.*;

public class ClusterList {
[Bindable]
        public var clusterList:ArrayCollection = new ArrayCollection();
        public var getUrl:String = "/clusterList/get";
        public var saveUrl:String = "/clusterList/save";
        public var updateUrl:String = "/clusterList/update";
        public var deployUrl:String = "/clusterList/deploy";
        public var getStatusUrl:String = "/clusterList/getStatus";
        public var getRpc:HTTPRpc;
        public var saveRpc:HTTPRpc;
        public var updateRpc:HTTPRpc;
        public var deployRpc:HTTPRpc;
        public var getStatusRpc:HTTPRpc;
                
        public function ClusterList() {
                getRpc = new HTTPRpc(getUrl);
                saveRpc = new HTTPRpc(saveUrl);
                updateRpc = new HTTPRpc(updateUrl);
                deployRpc = new HTTPRpc(deployUrl);
                getStatusRpc = new HTTPRpc(getStatusUrl);
                BindingUtils.bindSetter(function(list:Array):void{ clusterList = new ArrayCollection(list);}, getRpc, "result");
                // BindingUtils.bindSetter(updateStatus, getStatusRpc, "result");
                BindingUtils.bindSetter(updateComplete, updateRpc, "result");
                BindingUtils.bindSetter(deployComplete, deployRpc, "result");
                getRpc.call();
        }
        public function get():void {
                getRpc.call();
        }
        // public function getStatus():void {
        //         getStatusRpc.call();
        // }
        // public function updateStatus(status:Object):void {
        //         for each(var i:int in clusterList){
        //                         var info:Object;
        //                         if(!status.hasOwnProperty(clusterList[i].name))continue;
        //                         info = status[clusterList[i].name];
        //                         clusterList[i]['mpLabel']= info.mpLabel;
        //                         clusterList[i]['mpTotal']= info.mpTotal;
        //                         clusterList.itemUpdated(clusterList[i]);
        //                 }
        // }
        public function save():void {
                saveRpc.call({value:clusterList.source});
        }
        public function add(clusterName:String, numVMs:int):void {
                Log.traceCall('clusterList.add', arguments);
                clusterList.addItem({name:clusterName, numVMs:numVMs, mpLabel:'Not Availible.', mpTotal:-1});
                save();
        }
        public function del(index:int):void {
                Log.traceCall('clusterList.del', arguments);
                clusterList.removeItemAt(index);
                save();
        }
        public function makeItReady(index:int):void {
                var cluster:Object = clusterList.getItemAt(index);
                cluster['mpLabel'] = 'Peparing VM cluster...';
                cluster['mpTotal'] = 0;
                clusterList.itemUpdated(clusterList[index]);
                update(index);
        }
        public function reportStatus(cluster:Object):int {
                if(!(cluster && cluster.hasOwnProperty('mpLabel')))return -1;
                for (var i:int =0; i<clusterList.length; i++){
                                var info:Object;
                                if(cluster.name != clusterList[i].name)continue;
                                clusterList[i]['mpLabel']= cluster.mpLabel;
                                clusterList[i]['mpTotal']= cluster.mpTotal;
                                clusterList.itemUpdated(clusterList[i]);
                                return i;
                        }
                return -1;
        }
        public function updateComplete(cluster:Object):void {
                var i:int;
                i = reportStatus(cluster);
                if(i == -1)return;
                deploy(i);
        }
        public function deployComplete(cluster:Object):void {
                reportStatus(cluster);
        }
        public function update(index:int):void {
                Log.traceCall('clusterList.update', arguments);
                updateRpc.call({cluster:clusterList.getItemAt(index)});
        }
        public function deploy(index:int):void {
                Log.traceCall('clusterList.deploy', arguments);
                deployRpc.call({cluster:clusterList.getItemAt(index)});
        }
}
}