<template>
<div class=" flex flex-row gap-2 flex-wrap">
  <div class=" self-center h-fit">实体列表：</div>
  <div v-for="(tag, index) in tags" :key="tag+index" 
  :class='"entity entity-"+tag'>
    {{NERList[tag]}}
  </div>
</div>
</template>

<script setup>

const NERList = {
  "organization": "组织",
  "person": "人名",
  "location": "地名",
  "time": "时间",
  "thing": "事物",
  "metric": "度量",
}
// get key of NERList
const tags = Object.keys(NERList)

</script>