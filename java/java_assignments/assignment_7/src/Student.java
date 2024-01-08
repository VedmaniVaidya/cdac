import java.util.Comparator;
import java.util.Set;

public class Student implements Comparable<Student>{

    private int rollno;
    private String name;
    private double percentage;
    private Set<String> skillset;

    public Student(int rollno, String name, double percentage, Set<String> skillset) {
        this.rollno = rollno;
        this.name = name;
        this.percentage = percentage;
        this.skillset = skillset;
    }

    public int getRollno() {
        return rollno;
    }

    public String getName() {
        return name;
    }

    public double getPercentage() {
        return percentage;
    }

    public Set<String> getSkillset() {
        return skillset;
    }

    @Override
    public String toString() {
        return "Student [rollno=" + rollno + ", name=" + name + ", percentage=" + percentage + ", skillset=" + skillset
                + "]";
    }

    @Override
    public int compareTo(Student o) {

        if(this.getRollno() > o.getRollno()){
            return 1;
        } else if(this.getRollno() < o.getRollno()){
            return -1;
        }
        return 0;
    } 
    
}

class StudentComparatorByPercentage implements Comparator<Student>{

    @Override
    public int compare(Student o1, Student o2) {
          if(o1.getPercentage() < o2.getPercentage()){
            return 1;
        } else if(o1.getPercentage() > o2.getPercentage()){
            return -1;
        }
        return 0;
    }
    
}
