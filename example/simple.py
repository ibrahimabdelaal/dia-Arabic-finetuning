import soundfile as sf

from dia.model import Dia


model = Dia.from_pretrained("/home/ubuntu/work/dia-finetuning/.cpkts/dia_finetune_cv /ckpt_step6000.pth")

text = "ar: لِتَقْيِيمِ الأَدَاءِ وَدِقَّةِ وُسُومِ اللُّغَةِ بِشَكْلٍ كَامِلٍ، تَحْتَوِي هَذِهِ الجُمْلَةُ الاِخْتِبَارِيَّةُ عَلَى عِدَّةِ جُمَلٍ تَبَعِيَّةٍ، وَعَلَامَاتِ تَرْقِيمٍ مُخْتَلِفَةٍ، وَعَدَدٍ كَافٍ مِنَ الْكَلِمَاتِ."

output = model.generate(text)

sf.write("simple.mp3", output, 44100)
