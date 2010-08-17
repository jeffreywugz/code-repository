package mflex
{
import mx.containers.HBox;
import mx.controls.ProgressBar;
import mx.controls.Label;
import mx.controls.dataGridClasses.*;
import mflex.*;

public class MProgressBar extends HBox
//This should normally be extends ProgressBar, we are using HBox to have more control over the layout.
{
        private var pb:ProgressBar;
        private var msg:Label;
       
        public function MProgressBar() {
                //Create new ProgressBar Instance
                msg = new Label();
                msg.text = "Ready";
                pb = new ProgressBar();
                pb.label = "%3%%";
                //Set some layout things
                pb.indeterminate = true;
                pb.minimum=0;
                pb.maximum=100;
                pb.mode = "manual";
                pb.percentWidth = 50;         
                pb.labelPlacement = "center";
                this.setStyle("verticalAlign","middle");
                pb.visible = false;
                //Add ProgressBar as child
                addChild(msg);
                addChild(pb);
        }
       
        override public function set data(value:Object):void
                {
                        super.data = value;
                }  
       
        override protected function updateDisplayList(unscaledWidth:Number, unscaledHeight:Number) : void{
                msg.text = data.mpLabel;
                if((!data.hasOwnProperty("mpTotal")) || data.mpTotal == -1){
                        pb.visible = false;
                        pb.indeterminate=true;
                } else if(data.mpTotal == 0) {
                        pb.visible = true;
                        pb.indeterminate = true;
                        pb.mode = "event";
                        pb.label = "please wait...";
                } else {
                        pb.visible = true;
                        pb.label = "%3%%";
                        pb.indeterminate = false;
                        pb.setProgress(data.mpLoaded, data.mpTotal);
                }
                super.updateDisplayList(unscaledWidth, unscaledHeight);
        }
}
}