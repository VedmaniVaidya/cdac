package collectionDemo;

import java.util.List;
import java.util.Vector;
import java.util.*;

import javax.swing.text.html.HTMLDocument.Iterator;


//collection:-
// // List(Interface): Vector(class),ArrayList(Class), LinkedList(Class),PrimaryQueue(Class)
// // Queue(Interface):PrimaryQueue(Class)
// // Set(Interface): HashSet(Class), LinkedHashSet(C), TreeSet(Class)


//Map(Interface) doesnt extend from collection: HashMap, LinkedHashMao, TreeMap, HashTable



public class VectorDemo {
    public static void main(String[] args) {
        Vector<Integer> v1 = new Vector<>();
        List<Integer> v2 = new Vector<>();
        System.out.println(v1.size()+" "+v1.capacity());
        v1.add(10);
        v1.add(20);
        System.out.println(v1.size()+" "+v1.capacity());
        v1.add(30);
        v1.add(40);
        v1.add(50);
        System.out.println(v1.size()+" "+v1.capacity());
        System.out.println("Using toString");
        System.out.println(v1); //using toString method

        for(Integer val:v1){
            System.out.print(val+" ");
        }
        System.out.println();

        java.util.Iterator<Integer> itr = v1.iterator(); //using iterator
        System.out.println("Using iterator");
        while(itr.hasNext()){
            System.out.print(itr.next()+" ");
        }
    }
}
