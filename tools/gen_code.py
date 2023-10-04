fw = open("out.txt", "w")

snippet = """
                nitems = sizeof(kv->value.{});
                fread(&kv->value.{}, 1, nitems, file);
                offset += nitems;
                printf("Value = %d\\n", kv->value.{});
                printf("Current offset = %d; actual file offset = %d\\n", offset, ftell(file));
                break;
"""

types = [
  "int32",
  "float32",
  "uint64",
  "int64",
  "float64",
  "bool_",
  "str"
]

for type in types:
  fw.write("            {" + snippet.format(type, type, type) + "            }\n")

fw.close()
