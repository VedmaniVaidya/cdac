package Q1;

public class App {
    public static void main(String[] args) {
        CktPlayer c1 = new CktPlayer("Ramesh", 100);
        FtPlayer f1 = new FtPlayer("Suresh", 2);

        c1.printDetails();
        f1.printDetails();
    }
}

interface printable {
public void printDetails();
}

class CktPlayer implements printable{
    String name;
    int runs;

    public CktPlayer() {
        super();
        this.name="John Doe";
        this.runs=0;
    }


    public CktPlayer(String name, int runs) {
        this.name = name;
        this.runs = runs;
    }



    @Override
    public String toString() {
        return "CktPlayer [name=" + name + ", runs=" + runs + "]";
    }


    @Override
    public void printDetails() {
        System.out.println(this.toString());
    }

}

class FtPlayer implements printable{

    String name;
    int goals;

    public FtPlayer() {
        this.name = "John Doe";
        this.goals = 0;
    }

    
    public FtPlayer(String name, int goals) {
        this.name = name;
        this.goals = goals;
    }

    

    @Override
    public String toString() {
        return "FtPlayer [name=" + name + ", goals=" + goals + "]";
    }


    @Override
    public void printDetails() {
        System.out.println(this.toString());
    }

}