# Open Kidney Dataset

## Preparation

1. Request access to the dataset by completing the form on the [Open Kidney Dataset website](https://rsingla.ca/kidneyUS/).
2. Clone [the GitHub repository](https://github.com/rsingla92/kidneyUS/tree/main) containing the annotations:

```shell
git clone https://github.com/rsingla92/kidneyUS.git
```

3. Once you have access, download the ZIP file containing the images ("kidneyUS_images_14_june_2022.zip") and unzip the images into an "images" folder in your cloned repository:

```shell
unzip kidneyUS_images_14_june_2022.zip -d /path/to/kidneyUS/images
```

4. Now load it, letting UltraBench preprocess and cache it for you on first use:

```python
import ultrabench

data = ultrabench.load_dataset(name="OpenKidney", path="/path/to/kidneyUS")
```
