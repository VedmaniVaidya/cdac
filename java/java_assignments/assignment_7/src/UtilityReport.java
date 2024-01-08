import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class UtilityReport {

    private Map<String,Double> map;

    public UtilityReport() {
        map = new HashMap<>();
    }

    public Map<String, Double> getMap() {
        return map;
    }

    public void showReport(ArrayList<Student> students){
        if(students.isEmpty()){
            System.out.println("please enter data first");
            return;
        }
        for(int i=0;i<students.size();i++){
            map.put(students.get(i).getName(),students.get(i).getPercentage());
        }
        for(String s:map.keySet()){
           System.out.println(s + " -->  " +map.get(s));
        }
    }

    
    
}
