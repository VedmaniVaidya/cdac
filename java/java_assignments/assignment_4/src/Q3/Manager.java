package Q3;

public class Manager extends Employee{
    private float al_petrol;
    private float al_food;
    private float al_other;

   public Manager(String name , float basic_salary){
       super(name,basic_salary);
       this.al_petrol = 0.08f*basic_salary;
       this.al_food = 0.12f*basic_salary;
       this.al_other = 0.04f*basic_salary;
   }

    public float gross_salary(){
        return super.getSalary() + (this.al_petrol + this.al_food+this.al_other);
    }

    public float net_salary(){
        float pf = 0.125f * super.getSalary();
        return this.gross_salary() - pf;
    }

    public String display(){
        return super.toString() + " gross_salary: "+this.gross_salary()+ " net_salary: "+this.net_salary()+"]";
    }

  
    

}