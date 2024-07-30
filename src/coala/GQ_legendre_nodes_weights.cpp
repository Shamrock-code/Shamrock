//-------------------------------------------
// GQ legendre nodes and weights to order 20 
//-------------------------------------------
#include "options.hpp"
#include <iostream>
#include "GQ_legendre_nodes_weights.hpp"


void GQLeg_nodes(unsigned int k, flt* vecnodes){
	
	switch (k) {
	   case 1:
	      vecnodes[0] = ((flt)0);
	      break;
	   case 2:
	      vecnodes[0] = ((flt)-0.5773502691896258); vecnodes[1] = ((flt)0.5773502691896258);
	      break;  
	   case 3:
	      vecnodes[0] = ((flt)-0.7745966692414834); vecnodes[1] = ((flt)0.); vecnodes[2] = ((flt)0.7745966692414834);
	      break;
	   case 4:
	      vecnodes[0] = ((flt)-0.8611363115940526); vecnodes[1] = ((flt)-0.3399810435848563); vecnodes[2] = ((flt)0.3399810435848563); vecnodes[3] = ((flt)0.8611363115940526);
			break;
	   case 5:
	      vecnodes[0] = ((flt)-0.906179845938664); vecnodes[1] = ((flt)-0.5384693101056831); vecnodes[2] = ((flt)0.); vecnodes[3] = ((flt)0.5384693101056831); vecnodes[4] = ((flt)0.906179845938664);
			break;
	   case 6:
	      vecnodes[0] = ((flt)-0.932469514203152); vecnodes[1] = ((flt)-0.6612093864662645); vecnodes[2] = ((flt)-0.2386191860831969); vecnodes[3] = ((flt)0.2386191860831969); vecnodes[4] = ((flt)0.6612093864662645);
	      vecnodes[5] = ((flt)0.932469514203152);
			break;
	   case 7:
	      vecnodes[0] = ((flt)-0.949107912342759); vecnodes[1] = ((flt)-0.7415311855993944); vecnodes[2] = ((flt)-0.4058451513773972); vecnodes[3] = ((flt)0.); vecnodes[4] = ((flt)0.4058451513773972);
	      vecnodes[5] = ((flt)0.7415311855993944); vecnodes[6] = ((flt)0.949107912342759);
			break;
	   case 8:
	      vecnodes[0] = ((flt)-0.960289856497536); vecnodes[1] = ((flt)-0.7966664774136267); vecnodes[2] = ((flt)-0.525532409916329); vecnodes[3] = ((flt)-0.1834346424956498); vecnodes[4] = ((flt)0.1834346424956498);
	      vecnodes[5] = ((flt)0.525532409916329); vecnodes[6] = ((flt)0.7966664774136267); vecnodes[7] = ((flt)0.960289856497536);
			break;
	   case 9:
	      vecnodes[0] = ((flt)-0.968160239507626); vecnodes[1] = ((flt)-0.8360311073266358); vecnodes[2] = ((flt)-0.6133714327005904); vecnodes[3] = ((flt)-0.3242534234038089); vecnodes[4] = ((flt)0.);
	      vecnodes[5] = ((flt)0.3242534234038089); vecnodes[6] = ((flt)0.6133714327005904); vecnodes[7] = ((flt)0.8360311073266358); vecnodes[8] = ((flt)0.968160239507626);
			break;
	   case 10:
	      vecnodes[0] = ((flt)-0.973906528517172); vecnodes[1] = ((flt)-0.8650633666889845); vecnodes[2] = ((flt)-0.6794095682990244); vecnodes[3] = ((flt)-0.4333953941292472); vecnodes[4] = ((flt)-0.1488743389816312);
	      vecnodes[5] = ((flt)0.1488743389816312); vecnodes[6] = ((flt)0.4333953941292472); vecnodes[7] = ((flt)0.6794095682990244); vecnodes[8] =  ((flt)0.8650633666889845); vecnodes[9] = ((flt)0.973906528517172);
			break;
	   case 11:
	      vecnodes[0] = ((flt)-0.978228658146057); vecnodes[1] = ((flt)-0.8870625997680953); vecnodes[2] = ((flt)-0.7301520055740493); vecnodes[3] = ((flt)-0.5190961292068118); vecnodes[4] = ((flt)-0.269543155952345);
			vecnodes[5] = ((flt)0.); vecnodes[6] = ((flt)0.269543155952345); vecnodes[7] = ((flt)0.5190961292068118); vecnodes[8] = ((flt)0.7301520055740493); vecnodes[9] = ((flt)0.8870625997680953);
			vecnodes[10] = ((flt)0.978228658146057);
			break;
	   case 12:
	      vecnodes[0] = ((flt)-0.981560634246719); vecnodes[1] = ((flt)-0.904117256370475); vecnodes[2] = ((flt)-0.7699026741943047); vecnodes[3] = ((flt)-0.5873179542866174); vecnodes[4] = ((flt)-0.3678314989981802);
	      vecnodes[5] = ((flt)-0.1252334085114689); vecnodes[6] = ((flt)0.1252334085114689); vecnodes[7] = ((flt)0.3678314989981802); vecnodes[8] = ((flt)0.5873179542866174); vecnodes[9] = ((flt)0.7699026741943047);
	      vecnodes[10] = ((flt)0.904117256370475); vecnodes[11] = ((flt)0.981560634246719);
			break;
	   case 13:
	      vecnodes[0] = ((flt)-0.984183054718588); vecnodes[1] = ((flt)-0.917598399222978); vecnodes[2] = ((flt)-0.8015780907333099); vecnodes[3] = ((flt)-0.6423493394403402); vecnodes[4] = ((flt)-0.4484927510364469);
	      vecnodes[5] = ((flt)-0.2304583159551348); vecnodes[6] =((flt)0.); vecnodes[7] =((flt)0.2304583159551348); vecnodes[8] = ((flt)0.4484927510364469); vecnodes[9] = ((flt)0.6423493394403402);
	      vecnodes[10] = ((flt)0.8015780907333099); vecnodes[11] = ((flt)0.917598399222978); vecnodes[12] = ((flt)0.984183054718588);
			break;
	   case 14:
	      vecnodes[0] = ((flt)-0.986283808696812); vecnodes[1] =  ((flt)-0.928434883663574); vecnodes[2] =  ((flt)-0.827201315069765); vecnodes[3] =  ((flt)-0.6872929048116855); vecnodes[4] =  ((flt)-0.5152486363581541);
	      vecnodes[5] =  ((flt)-0.3191123689278898); vecnodes[6] =  ((flt)-0.1080549487073437); vecnodes[7] = ((flt)0.1080549487073437); vecnodes[8] =  ((flt)0.3191123689278898); vecnodes[9] =  ((flt)0.5152486363581541);
	      vecnodes[10] =  ((flt)0.6872929048116855); vecnodes[11] =  ((flt)0.827201315069765); vecnodes[12] =  ((flt)0.928434883663574); vecnodes[13] =  ((flt)0.986283808696812);
			break;
	   case 15:
	      vecnodes[0] = ((flt)-0.987992518020485); vecnodes[1] = ((flt)-0.937273392400706); vecnodes[2] = ((flt)-0.8482065834104272); vecnodes[3] = ((flt)-0.72441773136017); vecnodes[4] = ((flt)-0.5709721726085388);
			vecnodes[5] =((flt)-0.3941513470775634); vecnodes[6] = ((flt)-0.2011940939974345); vecnodes[7] = ((flt)0.); vecnodes[8] =((flt)0.2011940939974345); vecnodes[9] = ((flt)0.3941513470775634);
			vecnodes[10] =((flt)0.5709721726085388); vecnodes[11] = ((flt)0.72441773136017); vecnodes[12] = ((flt)0.8482065834104272); vecnodes[13] = ((flt)0.937273392400706); vecnodes[14] = ((flt)0.987992518020485);
			break;
	   case 16:
	      vecnodes[0] = ((flt)-0.98940093499165); vecnodes[1] = ((flt)-0.944575023073233); vecnodes[2] = ((flt)-0.8656312023878317); vecnodes[3] = ((flt)-0.755404408355003); vecnodes[4] = ((flt)-0.6178762444026437);
			vecnodes[5] =((flt)-0.4580167776572274); vecnodes[6] = ((flt)-0.2816035507792589); vecnodes[7] = ((flt)-0.0950125098376374); vecnodes[8] =((flt)0.0950125098376374); vecnodes[9] = ((flt)0.2816035507792589);
			vecnodes[10] =((flt)0.4580167776572274); vecnodes[11] = ((flt)0.6178762444026437); vecnodes[12] = ((flt)0.755404408355003); vecnodes[13] = ((flt)0.8656312023878317); vecnodes[14] = ((flt)0.944575023073233);
			vecnodes[15] =((flt)0.98940093499165);
			break;
	   case 17:
	      vecnodes[0] = ((flt)-0.990575475314417); vecnodes[1] = ((flt)-0.950675521768768); vecnodes[2] = ((flt)-0.8802391537269859); vecnodes[3] = ((flt)-0.7815140038968014); vecnodes[4] = ((flt)-0.6576711592166908);
			vecnodes[5] =((flt)-0.512690537086477); vecnodes[6] = ((flt)-0.3512317634538763); vecnodes[7] = ((flt)-0.1784841814958479); vecnodes[8] = ((flt)0.); vecnodes[9] =((flt)0.1784841814958479);
			vecnodes[10] =((flt)0.3512317634538763); vecnodes[11] = ((flt)0.512690537086477); vecnodes[12] = ((flt)0.6576711592166908); vecnodes[13] = ((flt)0.7815140038968014); vecnodes[14] = ((flt)0.8802391537269859);
			vecnodes[15] =((flt)0.950675521768768); vecnodes[16] = ((flt)0.990575475314417);
			break;
		case 18:
	      vecnodes[0] = ((flt)-0.991565168420931); vecnodes[1] = ((flt)-0.955823949571398); vecnodes[2] = ((flt)-0.8926024664975557); vecnodes[3] = ((flt)-0.8037049589725231); vecnodes[4] = ((flt)-0.6916870430603532);
	      vecnodes[5] =((flt)-0.5597708310739475); vecnodes[6] = ((flt)-0.4117511614628426); vecnodes[7] = ((flt)-0.2518862256915055); vecnodes[8] = ((flt)-0.0847750130417353); vecnodes[9] =((flt)0.0847750130417353);
	      vecnodes[10] =((flt)0.2518862256915055); vecnodes[11] = ((flt)0.4117511614628426); vecnodes[12] = ((flt)0.5597708310739475); vecnodes[13] = ((flt)0.6916870430603532); vecnodes[14] = ((flt)0.8037049589725231);
	      vecnodes[15] =((flt)0.8926024664975557); vecnodes[16] = ((flt)0.955823949571398); vecnodes[17] = ((flt)0.991565168420931);
			break;
	   case 19:
	      vecnodes[0] = ((flt)-0.992406843843584); vecnodes[1] = ((flt)-0.96020815213483); vecnodes[2] = ((flt)-0.903155903614818); vecnodes[3] = ((flt)-0.8227146565371428); vecnodes[4] = ((flt)-0.7209661773352294);
	      vecnodes[5] = ((flt)-0.600545304661681); vecnodes[6] = ((flt)-0.4645707413759609); vecnodes[7] = ((flt)-0.3165640999636298); vecnodes[8] = ((flt)-0.1603586456402254); vecnodes[9] = ((flt)0.);
			vecnodes[10] =((flt)0.1603586456402254); vecnodes[11] = ((flt)0.3165640999636298); vecnodes[12] = ((flt)0.4645707413759609); vecnodes[13] = ((flt)0.600545304661681); vecnodes[14] = ((flt)0.7209661773352294); 
			vecnodes[15] = ((flt)0.8227146565371428); vecnodes[16] =((flt)0.903155903614818); vecnodes[17] = ((flt)0.96020815213483); vecnodes[18] = ((flt)0.992406843843584);
			break;
	   case 20:
	      vecnodes[0] = ((flt)-0.993128599185095); vecnodes[1] = ((flt)-0.963971927277914); vecnodes[2] = ((flt)-0.912234428251326); vecnodes[3] = ((flt)-0.8391169718222188); vecnodes[4] = ((flt)-0.7463319064601508);
	      vecnodes[5] = ((flt)-0.636053680726515); vecnodes[6] = ((flt)-0.5108670019508271); vecnodes[7] = ((flt)-0.3737060887154196); vecnodes[8] = ((flt)-0.2277858511416451); vecnodes[9] = ((flt)-0.07652652113349733);
	      vecnodes[10] =((flt)0.07652652113349733); vecnodes[11] = ((flt)0.2277858511416451); vecnodes[12] = ((flt)0.3737060887154196); vecnodes[13] = ((flt)0.5108670019508271); vecnodes[14] = ((flt)0.636053680726515);
	      vecnodes[15] =((flt)0.7463319064601508); vecnodes[16] = ((flt)0.8391169718222188); vecnodes[17] = ((flt)0.912234428251326); vecnodes[18] = ((flt)0.963971927277914); vecnodes[19] = ((flt)0.993128599185095);
			break;
	   
	   default:
	      std::cerr<<"GQ_legendre_nodes_weights.cpp ->  GQLeg_nodes, only for k<=20"<<std::endl;
			// exit(-1);
	}
}


void GQLeg_weights(unsigned int k,flt* vecweights){
	
	switch (k){
	   case 1:
	      vecweights[0] = ((flt)2.);
	      break;
	   case 2:
	      vecweights[0] = ((flt)1.); vecweights[1] = ((flt)1.); 
	      break;
	   case 3:
	      vecweights[0] = ((flt)0.5555555555555557); vecweights[1] =((flt)0.8888888888888889); vecweights[2] =((flt)0.5555555555555557);
			break;
	   case 4:
	      vecweights[0] = ((flt)0.3478548451374538); vecweights[1] =((flt)0.6521451548625461); vecweights[2] =((flt)0.6521451548625461); vecweights[3] =((flt)0.3478548451374538);
			break;
	   case 5:
	      vecweights[0] = ((flt)0.2369268850561891); vecweights[1] =((flt)0.4786286704993665); vecweights[2] =((flt)0.5688888888888889); vecweights[3] =((flt)0.4786286704993665); vecweights[4] =((flt)0.2369268850561891);
			break;
		case 6:
	      vecweights[0] = ((flt)0.1713244923791703); vecweights[1] =((flt)0.3607615730481386); vecweights[2] =((flt)0.467913934572691); vecweights[3] =((flt)0.467913934572691); vecweights[4] =((flt)0.3607615730481386);
	      vecweights[5] =((flt)0.1713244923791703);
			break;
	   case 7:
	      vecweights[0] = ((flt)0.1294849661688697); vecweights[1] =((flt)0.2797053914892767); vecweights[2] =((flt)0.3818300505051189); vecweights[3] =((flt)0.4179591836734694); vecweights[4] =((flt)0.3818300505051189);
	      vecweights[5] =((flt)0.2797053914892767); vecweights[6] =((flt)0.1294849661688697);
			break;
	   case 8:
	      vecweights[0] = ((flt)0.1012285362903763); vecweights[1] = ((flt)0.2223810344533745); vecweights[2] = ((flt)0.3137066458778873); vecweights[3] = ((flt)0.362683783378362); vecweights[4] = ((flt)0.362683783378362);
	      vecweights[5] = ((flt)0.3137066458778873); vecweights[6] = ((flt)0.2223810344533745); vecweights[7] =((flt)0.1012285362903763);
			break;
	   case 9:
	      vecweights[0] = ((flt)0.08127438836157441); vecweights[1] = ((flt)0.1806481606948574); vecweights[2] = ((flt)0.2606106964029355); vecweights[3] = ((flt)0.3123470770400028); vecweights[4] = ((flt)0.3302393550012598);
	      vecweights[5] = ((flt)0.3123470770400028); vecweights[6] = ((flt)0.2606106964029355); vecweights[7] = ((flt)0.1806481606948574); vecweights[8] = ((flt)0.08127438836157441);
			break;
	   case 10:
	      vecweights[0] = ((flt)0.06667134430868814); vecweights[1] = ((flt)0.1494513491505806); vecweights[2] = ((flt)0.219086362515982); vecweights[3] = ((flt)0.2692667193099964); vecweights[4] = ((flt)0.2955242247147529);
	      vecweights[5] = ((flt)0.2955242247147529); vecweights[6] = ((flt)0.2692667193099964); vecweights[7] = ((flt)0.219086362515982); vecweights[8] = ((flt)0.1494513491505806); vecweights[9] = ((flt)0.06667134430868814);
			break;
	   case 11:
	      vecweights[0] = ((flt)0.05566856711617367); vecweights[1] = ((flt)0.1255803694649046); vecweights[2] = ((flt)0.1862902109277343); vecweights[3] = ((flt)0.2331937645919905); vecweights[4] = ((flt)0.2628045445102467);
	      vecweights[5] = ((flt)0.2729250867779006); vecweights[6] = ((flt)0.2628045445102467); vecweights[7] = ((flt)0.2331937645919905); vecweights[8] = ((flt)0.1862902109277343); vecweights[9] = ((flt)0.1255803694649046);
	      vecweights[10] = ((flt)0.05566856711617367);
			break;
	   case 12:
	      vecweights[0] = ((flt)0.04717533638651183); vecweights[1] = ((flt)0.1069393259953184); vecweights[2] = ((flt)0.1600783285433462); vecweights[3] = ((flt)0.2031674267230659); vecweights[4] = ((flt)0.2334925365383548);
	      vecweights[5] = ((flt)0.2491470458134028); vecweights[6] = ((flt)0.2491470458134028); vecweights[7] = ((flt)0.2334925365383548); vecweights[8] = ((flt)0.2031674267230659); vecweights[9] = ((flt)0.1600783285433462);
	      vecweights[10] = ((flt)0.1069393259953184); vecweights[11] = ((flt)0.04717533638651183);
			break;
	   case 13:
	      vecweights[0] = ((flt)0.04048400476531588); vecweights[1] = ((flt)0.0921214998377284); vecweights[2] = ((flt)0.1388735102197872); vecweights[3] = ((flt)0.1781459807619457); vecweights[4] = ((flt)0.2078160475368885);
	      vecweights[5] = ((flt)0.2262831802628972); vecweights[6] = ((flt)0.2325515532308739); vecweights[7] = ((flt)0.2262831802628972); vecweights[8] = ((flt)0.2078160475368885); vecweights[9] = ((flt)0.1781459807619457);
	      vecweights[10] = ((flt)0.1388735102197872); vecweights[11] = ((flt)0.0921214998377284); vecweights[12] = ((flt)0.04048400476531588);
			break;
	   case 14:
	      vecweights[0] = ((flt)0.03511946033175186); vecweights[1] = ((flt)0.08015808715976021); vecweights[2] = ((flt)0.1215185706879032); vecweights[3] = ((flt)0.1572031671581935); vecweights[4] = ((flt)0.1855383974779378);
	      vecweights[5] = ((flt)0.2051984637212956); vecweights[6] = ((flt)0.2152638534631578); vecweights[7] = ((flt)0.2152638534631578); vecweights[8] = ((flt)0.2051984637212956); vecweights[9] = ((flt)0.1855383974779378);
	      vecweights[10] = ((flt)0.1572031671581935); vecweights[11] = ((flt)0.1215185706879032); vecweights[12] = ((flt)0.08015808715976021); vecweights[13] = ((flt)0.03511946033175186);
			break;
	   case 15:
	      vecweights[0] = ((flt)0.03075324199611727); vecweights[1] = ((flt)0.07036604748810812); vecweights[2] = ((flt)0.1071592204671719); vecweights[3] = ((flt)0.1395706779261543); vecweights[4] = ((flt)0.1662692058169939);
	      vecweights[5] = ((flt)0.1861610000155622); vecweights[6] = ((flt)0.1984314853271116); vecweights[7] = ((flt)0.2025782419255613); vecweights[8] = ((flt)0.1984314853271116); vecweights[9] = ((flt)0.1861610000155622);
	      vecweights[10] = ((flt)0.1662692058169939); vecweights[11] = ((flt)0.1395706779261543); vecweights[12] = ((flt)0.1071592204671719); vecweights[13] = ((flt)0.07036604748810812); vecweights[14] = ((flt)0.03075324199611727);
			break;
	   case 16:
	      vecweights[0] = ((flt)0.02715245941175409); vecweights[1] = ((flt)0.06225352393864789); vecweights[2] = ((flt)0.0951585116824928); vecweights[3] = ((flt)0.1246289712555339); vecweights[4] = ((flt)0.1495959888165767);
	      vecweights[5] = ((flt)0.1691565193950025); vecweights[6] = ((flt)0.1826034150449236); vecweights[7] = ((flt)0.1894506104550685); vecweights[8] = ((flt)0.1894506104550685); vecweights[9] = ((flt)0.1826034150449236);
	      vecweights[10] = ((flt)0.1691565193950025); vecweights[11] = ((flt)0.1495959888165767); vecweights[12] = ((flt)0.1246289712555339); vecweights[13] = ((flt)0.0951585116824928); vecweights[14] = ((flt)0.06225352393864789);
	      vecweights[15] = ((flt)0.02715245941175409);
			break;
	   case 17:
	      vecweights[0] = ((flt)0.02414830286854793); vecweights[1] = ((flt)0.0554595293739872); vecweights[2] = ((flt)0.08503614831717918); vecweights[3] = ((flt)0.111883847193404); vecweights[4] = ((flt)0.1351363684685255);
	      vecweights[5] = ((flt)0.1540457610768103); vecweights[6] = ((flt)0.16800410215645); vecweights[7] = ((flt)0.1765627053669926); vecweights[8] = ((flt)0.1794464703562065); vecweights[9] = ((flt)0.1765627053669926);
	      vecweights[10] = ((flt)0.16800410215645); vecweights[11] = ((flt)0.1540457610768103); vecweights[12] = ((flt)0.1351363684685255); vecweights[13] = ((flt)0.111883847193404); vecweights[14] = ((flt)0.08503614831717918);
	      vecweights[15] = ((flt)0.0554595293739872); vecweights[16] = ((flt)0.02414830286854793);
			break;
	   case 18:
	      vecweights[0] = ((flt)0.02161601352648331); vecweights[1] = ((flt)0.0497145488949698); vecweights[2] = ((flt)0.07642573025488906); vecweights[3] = ((flt)0.1009420441062872); vecweights[4] = ((flt)0.1225552067114785);
	      vecweights[5] = ((flt)0.1406429146706507); vecweights[6] = ((flt)0.1546846751262652); vecweights[7] = ((flt)0.1642764837458327); vecweights[8] = ((flt)0.1691423829631436); vecweights[9] = ((flt)0.1691423829631436);
	      vecweights[10] = ((flt)0.1642764837458327); vecweights[11] = ((flt)0.1546846751262652); vecweights[12] = ((flt)0.1406429146706507); vecweights[13] = ((flt)0.1225552067114785); vecweights[14] = ((flt)0.1009420441062872);
	      vecweights[15] = ((flt)0.07642573025488906); vecweights[16] = ((flt)0.0497145488949698); vecweights[17] = ((flt)0.02161601352648331);
			break;
	   case 19:
	      vecweights[0] = ((flt)0.01946178822972648); vecweights[1] = ((flt)0.0448142267656996); vecweights[2] = ((flt)0.06904454273764123); vecweights[3] = ((flt)0.09149002162245); vecweights[4] = ((flt)0.111566645547334);
	      vecweights[5] = ((flt)0.1287539625393362); vecweights[6] = ((flt)0.1426067021736066); vecweights[7] = ((flt)0.1527660420658597); vecweights[8] = ((flt)0.1589688433939543); vecweights[9] = ((flt)0.1610544498487837);
	      vecweights[10] = ((flt)0.1589688433939543); vecweights[11] = ((flt)0.1527660420658597); vecweights[12] = ((flt)0.1426067021736066); vecweights[13] = ((flt)0.1287539625393362); vecweights[14] = ((flt)0.111566645547334);
	      vecweights[15] = ((flt)0.09149002162245); vecweights[16] = ((flt)0.06904454273764123); vecweights[17] = ((flt)0.0448142267656996); vecweights[18] = ((flt)0.01946178822972648);
			break;
	   case 20:
	      vecweights[0] = ((flt)0.01761400713915212); vecweights[1] = ((flt)0.04060142980038694); vecweights[2] = ((flt)0.06267204833410906); vecweights[3] = ((flt)0.08327674157670475); vecweights[4] = ((flt)0.1019301198172404);
	      vecweights[5] = ((flt)0.1181945319615184); vecweights[6] = ((flt)0.1316886384491766); vecweights[7] = ((flt)0.1420961093183821); vecweights[8] = ((flt)0.1491729864726037); vecweights[9] = ((flt)0.1527533871307259);
	      vecweights[10] = ((flt)0.1527533871307259); vecweights[11] = ((flt)0.1491729864726037); vecweights[12] = ((flt)0.1420961093183821); vecweights[13] = ((flt)0.1316886384491766); vecweights[14] = ((flt)0.1181945319615184);
	      vecweights[15] = ((flt)0.1019301198172404); vecweights[16] = ((flt)0.08327674157670475); vecweights[17] = ((flt)0.06267204833410906); vecweights[18] = ((flt)0.04060142980038694); vecweights[19] =((flt)0.01761400713915212);
			break;
	   default:
	      std::cerr<<"GQ_legendre_nodes_weights.cpp ->  GQLeg_weights, only for k<=20"<<std::endl;
			// exit(-1);
	}


}
