package genericsDemo;

//generic class
class Main<G>{
    public void swap(G a, G b){
        G temp = a;
        a = b;
        b = temp;
        System.out.println(a+" "+b);
    }
}

public class Swap {
    public static void main(String[] args) {
        Main<Integer> a = new Main();
        a.swap(20, 10);
        Main<String> b = new Main();
        b.swap("Hello", "World");
    }
}
