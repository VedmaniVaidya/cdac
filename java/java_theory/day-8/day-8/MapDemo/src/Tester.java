import java.util.*;


public class Tester {
	public static void main(String[] args) {
		
		Map<Integer, String> map=new HashMap<>();
		
		map.put(101, "Maths");
		map.put(105, "Science");
		map.put(104, "English");
		map.put(null, null);
		
		System.out.println(map);
		
		Set<Integer> s=map.keySet();
		for(Integer key:s)
		{
			System.out.println(key+"   "+map.get(key));
		}
	}
}
