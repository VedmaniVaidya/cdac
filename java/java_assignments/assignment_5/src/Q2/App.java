package Q2;

public class App{
    public static void main(String[] args) {
        Vehicle v1 = new Vehicle("MH12RW2048", "BMW", 1000000);
        try {  
        Vehicle v2 = (Vehicle)v1.clone();//This create a deep copy of object
        System.out.println(v2);
        System.out.println(v2.hashCode());
        }catch (Exception e){
            System.out.println(e);
        }
        System.out.println(v1);
        Vehicle v3 = v1;// This create the reference to the object
        System.out.println(v1.hashCode());
        System.out.println(v3.hashCode());
        //v1 and v3 point to same memory
    }
}


class Vehicle implements Cloneable{
    String vehicleNumber;
    String vehicleName;
    float vehiclePrice;


    public Vehicle() {
        this.vehicleNumber = null;
        this.vehicleName = null;
        this.vehiclePrice = 0;
    }

    public Vehicle(String vehicleNumber, String vehicleName, float vehiclePrice) {
        this.vehicleNumber = vehicleNumber;
        this.vehicleName = vehicleName;
        this.vehiclePrice = vehiclePrice;
    }


    @Override
    public String toString() {
        return "Vehicle [vehicleNumber=" + vehicleNumber + ", vehicleName=" + vehicleName + ", vehiclePrice="
                + vehiclePrice + "]";
    }

    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }

    
}
