import java.util.ArrayList;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeSet;

public class TestVehicle {
    public static void main(String[] args) {
        Set<Vehicle> vehicles = new TreeSet<>(); 
        Scanner sc = new Scanner(System.in);
        String choice = "-1";

        while (choice!="q"){
            System.out.println(" a.Add vehicle into the hashset\n" +
             "b. display details of all vehicles from hashset\n" +
             "c. dispaly details of maximum and minumum cost vehicle from hashset\n"+
             "q. exit"
              );
            choice = sc.next();

            switch(choice){
                case "a":
                    addVeicles(vehicles);
                case "b":
                    dispalyvehicles(vehicles);
                case "c":
                   max_min_cost(vehicles);
                default:

            }

        }

    }

   
    private static void addVeicles(Set<Vehicle> vehicles) {
        Scanner sc = new Scanner(System.in);
        vehicles.add(new Vehicle(sc.nextInt(),sc.next(), sc.nextDouble()));
    }


    private static void max_min_cost(Set<Vehicle> vehicles) {
        
        ArrayList<Vehicle> arr = new ArrayList<>(vehicles);
        System.out.println("vehicle with minumum cost");
        System.out.println(arr.get(0));
         System.out.println("vehicle with maximum cost");
        System.out.println(arr.get(arr.size()-1));
    }

     private static void dispalyvehicles(Set<Vehicle> vehicles) {
        for(Vehicle vh :vehicles){
            System.out.println(vh);
        }
    }



}
