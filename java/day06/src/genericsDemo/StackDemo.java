package genericsDemo;

class Stack<G>{
    private int size;
    private G [] arr;
    private int top;

    public Stack(int s) {
        super();
        this.size=s;
        this.arr=(G[])new Object[size];
        this.top=-1;
    }

    public void push(G val){
        if (!isFull()){
        this.arr[++top]=val;
        }
        else
        System.out.println("Overflow!");
    }

    public G pop(){
        if(!isEmpty())
        {
            return arr[top--];
        }
        else{
            System.out.println("Underflow!");
        return null;
        }
    }

    public boolean isFull(){
        return top==size-1;
    }

    public boolean isEmpty()
    {
        return top==-1;
    }

}

public class StackDemo {
    public static void main(String[] args) {
        Stack<Integer> s1 = new Stack<>(10);
        s1.push(1);
        s1.push(2);
        s1.push(3);
        s1.push(4);
        s1.push(10);
        s1.push(10);
        s1.push(10);
        s1.push(10);
        s1.push(10);
        s1.push(10);
        // s1.push(10);
        System.out.println(s1.pop());
        System.out.println(s1.pop());
        System.out.println(s1.pop());
        System.out.println(s1.pop());
        System.out.println(s1.pop());
        System.out.println(s1.pop());
        System.out.println(s1.pop());
        System.out.println(s1.pop());
        System.out.println(s1.pop());
        System.out.println(s1.pop());

        Stack<String> s2= new Stack<>(10);
        s2.push("Hello");
        s2.push("World");
        s2.push("Hello");
        s2.push("World");
        s2.push("Hello");
        s2.push("World");
        s2.push("Hello");
        s2.push("World");
        s2.push("Hello");
        s2.push("World");
        System.out.println(s2.pop());
        System.out.println(s2.pop());
        System.out.println(s2.pop());
        System.out.println(s2.pop());
        System.out.println(s2.pop());
        System.out.println(s2.pop());
        System.out.println(s2.pop());
        System.out.println(s2.pop());
        System.out.println(s2.pop());
        System.out.println(s2.pop());
    }
}
