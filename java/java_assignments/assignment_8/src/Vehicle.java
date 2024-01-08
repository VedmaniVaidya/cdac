public class Vehicle implements Comparable<Vehicle>{

    private int vehicle_no;
    private String vehicle_name;
    private double vehicle_cost;

    public Vehicle(int vehicle_no, String vehicle_name, double vehicle_cost) {
        this.vehicle_no = vehicle_no;
        this.vehicle_name = vehicle_name;
        this.vehicle_cost = vehicle_cost;
    }

    public int getVehicle_no() {
        return vehicle_no;
    }

    public String getVehicle_name() {
        return vehicle_name;
    }

    public double getVehicle_cost() {
        return vehicle_cost;
    }

    @Override
    public String toString() {
        return "Vehicle [vehicle_no=" + vehicle_no + ", vehicle_name=" + vehicle_name + ", vehicle_cost=" + vehicle_cost
                + "]";
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + vehicle_no;
        result = prime * result + ((vehicle_name == null) ? 0 : vehicle_name.hashCode());
        long temp;
        temp = Double.doubleToLongBits(vehicle_cost);
        result = prime * result + (int) (temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Vehicle other = (Vehicle) obj;
        if (vehicle_no != other.vehicle_no)
            return false;
        if (vehicle_name == null) {
            if (other.vehicle_name != null)
                return false;
        } else if (!vehicle_name.equals(other.vehicle_name))
            return false;
        if (Double.doubleToLongBits(vehicle_cost) != Double.doubleToLongBits(other.vehicle_cost))
            return false;
        return true;
    }

    @Override
    public int compareTo(Vehicle o) {
        if(this.getVehicle_cost()<o.getVehicle_cost()){
            return 1;
        } else if(this.getVehicle_cost() > o.getVehicle_cost()){
            return -1;
        }
        return 0;
    }

    

}
