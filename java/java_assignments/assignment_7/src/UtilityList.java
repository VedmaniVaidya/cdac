import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Scanner;


public class UtilityList {
    
    private ArrayList<Student> students;

    
        
    public UtilityList() {
        students = new ArrayList<>();
    }

    public  void sortByRollno(){
           Collections.sort(students);
           for(Student std:students){
            System.out.println(std);
           }
    }

    public void sortByPercentage(){
           Collections.sort(students,new StudentComparatorByPercentage());
           for(Student std:students){
            System.out.println(std);
           }
    }

    public void printlist() {
        for(Student st:students){
            System.out.println(st);
        }
    }

    public  void createList(){
        Scanner sc = new Scanner(System.in);
        
        for(int i = 0;i<5;i++){
            int rollno = sc.nextInt();
            sc.nextLine();
            String name = sc.nextLine();
            double percentage = sc.nextDouble();
            students.add(new Student(rollno, name, percentage, getHobbies()));
        }
      
    }

    private  HashSet<String> getHobbies() {
        HashSet<String> hobbies = new HashSet<>();
        Scanner sc = new Scanner(System.in);
        System.out.println("enter number of hobbies");
        int n = sc.nextInt(                               );
        for(int i = 0;i<n;i++){
           hobbies.add(sc.next());
        }
        return hobbies;
    }

    public ArrayList<Student> getStudents() {
        return students;
    }

}
