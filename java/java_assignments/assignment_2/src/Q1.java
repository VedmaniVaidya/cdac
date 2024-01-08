import java.util.Scanner;

public class Q1 {

	public static void main(String[] args) {
		
		Scanner sc = new Scanner(System.in);
		int[] arr = new int[5];
		
		for(int i=0;i<arr.length;i++) {
			arr[i] = sc.nextInt();
		}
		
		for(int val:arr)
			System.out.print(val+" ");
		System.out.println();
		
		max_min(arr);
		
		int[] new_array = array_operation(arr);
		for(int val:new_array)
			System.out.print(val+" ");
		

	}
	
	public static int[] array_operation(int[] nums) {
		int[] newarray = new int[nums.length];
		for (int i = 0;i<nums.length;i++) {
			newarray[i] = nums[i]*5 ;
		}
		return newarray;
	}
	
	public static void max_min(int[] arr) {
		
		int max = Integer.MIN_VALUE;
		int min = Integer.MAX_VALUE;
		
		for(int val:arr) {
			if(val>max) {
				max = val;
			}
			if(val<min) {
				min = val;
			}
		}
		System.out.println("Maximum:"+max);
		System.out.println("Minimum:"+min);
	}

}
