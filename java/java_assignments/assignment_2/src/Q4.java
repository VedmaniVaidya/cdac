//Write a method “add” to add n number of integers. Use VarArgs
public class Q4 {
    public static void main(String[] args){
        int sum = add(1,2,3,4,5,6,7,8,9,10);
        System.out.println(sum);
    }


    public static int add(int ...a){//...a is syntax for variable args like *args in python (tuple expansion)
        int sum=0;
        for(int i=0;i<a.length;i++){
            sum+=a[i];
        }
        return sum;

    }
}

