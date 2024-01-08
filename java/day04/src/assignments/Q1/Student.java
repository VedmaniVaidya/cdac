package assignments.Q1;

public class Student {
    private int student_rollno;
    private String name;
    private Date dob;
    static int cnt = 1;
    
    public Student(){
        student_rollno = cnt++;
        name = "John Doe";
        dob = new Date();
    }

    public Student(String name,int day,int month,int year){
         this.student_rollno = cnt++;
         this.name = name;
         this.dob = new Date(day,month,year);
    }

    public String show() {
        return "Student [student_rollno=" + student_rollno + ", name=" + name + ", dob=" + dob.show()+ "]";
    }

  
}
