package ent
{
import mflex.*;
public class Agent extends HTTPObjProxy {
        public static var url:String = "/agent";
        public function Agent() {
                super(url);
        }
}
}