import java.util.*;

public class Tester {

	public static void main(String[] args) 
	{
		Set<Integer> s=new TreeSet<>();
		s.add(13);
		s.add(10);
		s.add(123);
		s.add(13);
		s.add(1);
		s.add(15);
		s.add(15);
		
		System.out.println(s);
	}

}
