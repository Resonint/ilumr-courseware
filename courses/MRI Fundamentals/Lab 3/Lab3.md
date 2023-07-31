## Lab 3: Selective Excitation 

| #   | Topic                 | Status         |
| --- | --------------------- | -------------- |
| 3   | Selective Excitation  | In Development |

### Learning opportunities 

* Further cement understanding of the relationship between Time and Frequency domains
    * Demonstrate How shaped RF pulses can affect the BW of excitation
    * Relate the BW of excitation to the position inside the sample
* Deepen the concepts behind required calibration and pulse sequence setup
    * Re-iterate the requirements of what is needed to get an image:
      * Need to know:
        * Bo or resonance frequency 
        * Gradient calibration in units (max Strength and slew rate e.g. thinnest slice excitation possible)
        * B1 parameters for 90 and 180 pulses

### Outline
* Understanding excitation
  * Review Excitation using hard pulses
    * Get B1 in uT for a 90-degree hard pulse based on a given pulse width using $\theta = \int_0^T B_1(t)dt$
    * Use that to get a 90 flip angle for shorter or longer pulse widths
    * From the above RF calibration for B1 

* Introduction to slice selection
  * Excitation using a BW-limited pulse e.g. SINC or Gaussian shape in the presence of a gradient. Start with low power to ensure we are below 90o flip angle for excitation.
    * Change BW * gradient observe 
    * Change readout directions excite on Z and readout on x,y,z to understand spatial dependence.
  * Introduce slice offset
    * Change excitation frequency (fo) and see how readout changes which way the excitation slice moves

* Demonstrate Non-linearity of Bloch equations and excitations: 
  * Using a SINC RF pulse with a Z slice selection gradient,  keep increasing B1  amplitude of the RF pulses and observe a slice selection shape. The RF shape will look rectangular until 90 flip angle after 90 $\approx 105^o$. The distortion will be significant and observed as a dip in the center of the slice profile.
  *  

* Go through the Calibration steps of Gradients and B1 
  * Students should be able to convert system used values to physical Gradient and B1 values
  * Be able to excite the specific 
  * **This may need to be its own lab?** I feel like this is a good spot because slice selection can be easily tied to a physical measurement slice thickness. Knowing the conversion units will be useful for all sorts of other calculations e.g. FOV etc...

**Optional extra if time permits
* It would be great to introduce oblique excitation. By rotating the excitation plane 
  * 



**Questions:** 

How best to implement Calibration of shaped RF pulses?

Options:
1) Slow and reliable RF power sweep or Pulse Width modulation vs signal amplitude
2) Using the Stimulated Echo approach of echo size comparison?  a-a-a   a = 90 when  Stimulated Echo size = Echo size  
3) Using Bloch-Siegert approach with an RF-offresonance two-shot approach

Ref: I came across this in Canada when I visited Mike in Hamilton:  Bloch–Siegert B1 Mapping and calibration https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2933656/