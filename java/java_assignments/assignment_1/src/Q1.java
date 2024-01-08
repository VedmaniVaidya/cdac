import java.util.Scanner;

public class Q1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int totalSubjects = 5;
        double sum = 0;

        for(int i = 1; i <= totalSubjects; i++) {
            System.out.println("Enter marks for subject " + i + ": ");
            double marks = sc.nextDouble();
            sum += marks;
        }

        double average = sum / totalSubjects;
        System.out.println("The average marks are: " + average);
    }
}