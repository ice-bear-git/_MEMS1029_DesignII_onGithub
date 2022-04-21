# Brakes

Author: Ziang Cao

STILL NEED TO DO:
* Meanwhile the diameter of the hydrualic line. -- Just buy the meters.
* Find the material and friction for the Pads against the Hub disc.


## Text
### Introduction
> The very common rim brakes and drum brakes covered in the lecture are desinged with self-energization idea (`which can leverage the input force`). While this feature is important in reducing the braking effort required, it also has a disadvantage.
> When drum brakes are used as vehicle brakes, only a slight change in the coefficient of friction will cause a large change in pedal force required for braking. Then it would be terrible for braking during in the rainy day.  

To get rid of thouse issues, we switch to the disc brake. Thanks to the calipers which can be made very stiff indeed and the fact that a disc is much smaller than the rim on a bicycle wheel.  
There are relative more braking power applied to a slower linear velocity hard rotating area.
![More Powerful brake——1:29](https://github.com/ice-bear-git/_MEMS1029_DesignII_onGithub/blob/main/Project/Project3/Brakes/SizeOfDisc2.PNG)

Back to the fundamental, when you apply the pressure at hand-level paddel, the hydraulic line pass the pressure by fully sealed fluid and push pistons behind the pads at two calipers. As pushing against the rotation of the hubs, it slows down the hubs as well as the rear wheel. 
![BrakeLine](https://github.com/ice-bear-git/_MEMS1029_DesignII_onGithub/blob/main/Project/Project3/Brakes/BrakeLine.PNG)
![WhenItBrakes1](https://github.com/ice-bear-git/_MEMS1029_DesignII_onGithub/blob/main/Project/Project3/Brakes/WhenItBrakes1.PNG)


### Braking force analysis -- leverage
By `Pascal's Law`:  
External pressure applied to an eclosed incompressible fluid transmitted uniformly throughout the volume of the liquid.   
Hence, the leverage ratio is based on the dimension ratio of piston bore over the hydralic line diameter. 
![Pressure](https://github.com/ice-bear-git/_MEMS1029_DesignII_onGithub/blob/main/Project/Project3/Brakes/Pressure.PNG)
![WhenItBrakes2](https://github.com/ice-bear-git/_MEMS1029_DesignII_onGithub/blob/main/Project/Project3/Brakes/WhenItBrakes2.PNG)
Typically, the ratio is within **10-14**(currently guess).
Meanwhile, based on this [link](https://www.markwilliams.com/braketech.html)
, the typical force the rider will apply using their hands is `100 lbs ~= 444 N`.  


As the common dimension for the piston is found by this [link](https://www.zeckhausen.com/catalog/index.php?cPath=6446_6472). And the pressure calculatio use the `whole Boot Diameter`(I believe the outer piston wall is also moving), while the pad diameter for the braking using the piston diameter(assume pads have similar sizes).  
For example, when taking the Piston Size of `32 mm` with Dust Boot Diameter of `42 mm`. 
``` Python
Braking force = 444 N * (42/4) * (42/32) = ... N

```


### if there is water between the brake pad
Based on the material of the `brake`, the coefficient would be different.

We can gain some sense from the [two types of friction coefficient change between wheels and different surface](https://www.exploratorium.edu/cycling/brakes2.html)

| Surface | Adhesion coefficient |  Rolling coefficient | 
| --------------  | ---------- | ------- |  
| Dry Concrete		| 0.85 | 0.014 |  
| Wet Concrete		| 0.55 | 0.014 | 
| Sand		| 0.35 | 0.3 | 
| Ice		| 0.1 | 0.014 | 

I believe the case is most similar to Dry concrete surface to wet one. hence, to hold the same braking force, I would use `85/55 * 444N` for the hand input.  

For a typical biking speed of `8mi/hr`, the stopping distance would enlonger from `0.74 m` to `1.13 m`.


However, in the reality the disc brake performace beter than the above analysis. Thanks to the holes on its hub disc, which allows water and other fabricates to leave from the braking surface.
![Holes](https://github.com/ice-bear-git/_MEMS1029_DesignII_onGithub/blob/main/Project/Project3/Brakes/HolsOnDisc.PNG)
Meanwhile, the dics is out of the small water puddle on the ground.
![All weather——1:18](https://github.com/ice-bear-git/_MEMS1029_DesignII_onGithub/blob/main/Project/Project3/Brakes/AllWeather.PNG)



## Other Resource
1. Calculate Minimum Stopping Distance of a Bicyclist, [link](https://www.exploratorium.edu/cycling/brakes2.html)
* contains `Adhesion coefficient` and `Rolling coefficient` between the wheel and the surface

| Surface      	  | Adhesion coefficien |  Rolling coefficient | 
| --------------  | ---------- | ------- |  
| Dry Concrete		| 0.85 | 0.014 |  
| Wet Concrete		| 0.55 | 0.014 | 
| Sand		| 0.35 | 0.3 | 
| Ice		| 0.1 | 0.014 | 


2. Youtube: [How Disc Brake works?](https://www.youtube.com/watch?v=LKzLQUvVSOY)
3. Youtube: [How Do Disc Brakes Actually Work?](https://www.youtube.com/watch?v=U_Rnr_flVq8)
4. Beake Tech and FAQ [usually numbers](https://www.markwilliams.com/braketech.html)
5. SHIMANO: [R8000]
* Brake paddel: [link](https://bike.shimano.com/en-AU/product/component/ultegra-r8000/ST-R8020-L.html)
6. TextBook Chap16-6 + Example 16-4