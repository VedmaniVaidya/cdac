import java.util.Scanner;

public class Array1D {

	public static void main(String[] args) {
		int [] a = {10,20,30,40,50};
		int [] b = new int[] {5,4,3,2,1};
		int [] c = new int[7];
		
		Scanner sc = new Scanner(System.in);
		
		System.out.println("Enter array elements");
		
		for (int i=0;i<c.length;i++) {
			c[i] = sc.nextInt();
		}
		System.out.println("Array-");
		for (int i=0;i<c.length;i++)
			System.out.print(c[i]+" ");
		
		System.out.println();
//for loop to interate over array, this is read only loop, can be used only to display the contents of an array		
		for (int val:c)
			System.out.print(val+" ");

	}

}
