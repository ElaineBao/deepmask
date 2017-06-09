--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

Run full scene inference in sample image
------------------------------------------------------------------------------]]

require 'torch'
require 'cutorch'
require 'image'

function mysplit(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end
--------------------------------------------------------------------------------
-- parse arguments
local cmd = torch.CmdLine()
cmd:text()
cmd:text('evaluate deepmask/sharpmask')
cmd:text()
cmd:argument('-model', 'path to model to load')
cmd:text('Options:')
cmd:option('-imglist','/disk2/data/ILSVRC2017/ILSVRC/ImageSets/DET/val.txt', 'path/to/img/list')
cmd:option('-datapath','/disk2/data/ILSVRC2017/ILSVRC/Data/DET/val/','path/to/img/data')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-np', 500,'number of proposals to save in test')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf', .5, 'final scale')
cmd:option('-ss', 2., 'scale step')
cmd:option('-dm', false, 'use DeepMask version of SharpMask')

local config = cmd:parse(arg)
--------------------------------------------------------------------------------
-- various initializations
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(config.gpu)

local coco = require 'coco'
local maskApi = coco.MaskApi

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

--------------------------------------------------------------------------------
-- load moodel
paths.dofile('DeepMask.lua')
paths.dofile('SharpMask.lua')

print('| loading model file... ' .. config.model)
local m = torch.load(config.model..'/model.t7')
local model = m.model
model:inference(config.np)
model:cuda()

--------------------------------------------------------------------------------
-- create inference module
local scales = {}
for i = config.si,config.sf,config.ss do table.insert(scales,2^i) end

if torch.type(model)=='nn.DeepMask' then
  paths.dofile('InferDeepMask.lua')
elseif torch.type(model)=='nn.SharpMask' then
  paths.dofile('InferSharpMask.lua')
end

local infer = Infer{
  np = config.np,
  scales = scales,
  meanstd = meanstd,
  model = model,
  dm = config.dm,
}

--------------------------------------------------------------------------------
-- do it
print('| start')
local imglist = io.open(config.imglist)
local file = io.open("test.txt", "w+")
for line in io.lines(config.imglist) do
  local index = mysplit(line)
  index = index[1]
  local img_path = string.format("%s%s.JPEG",config.datapath , index)
  print("img_path:", img_path)

  -- load image
  local img = image.load(img_path)
  local h,w = img:size(2),img:size(3)
  local channel = img:size(1)
  print("h,w,channel:",h,w,channel)
  if (channel == 3) and (h < 1000) and (w < 1000) then

      -- forward all scales
      infer:forward(img)

      -- get top propsals
      local masks,scores = infer:getTopProps(.2,h,w)

      -- save result
      local res = img:clone()
      maskApi.drawMasks(res, masks, 10)
      local Rs = maskApi.encode( masks )
      local bbs  = maskApi.toBbox( Rs )
      local num_bb = bbs:size(1)
      local clr =  torch.rand(3)*.6+.4

      for i=1,num_bb do
        local score = scores[i][1]
        if score > 0.2 then
            local x1 = bbs[i][1]
            local y1 = bbs[i][2]
            local x2 = bbs[i][1] + bbs[i][3]
            local y2 = bbs[i][2] + bbs[i][4]
            --print('bbox score:',score)
            maskApi.drawLine( res, x1,y1,x1,y2, .75, clr)
            maskApi.drawLine( res, x1,y1,x2,y1, .75, clr)
            maskApi.drawLine( res, x1,y2,x2,y2, .75, clr)
            maskApi.drawLine( res, x2,y1,x2,y2, .75, clr)
            file:write(x1,' ',y1,' ',x2,' ',y2,' ',score,' ')
        end
      end
  end
  file:write('\n')
end
file:close()

--print(bbs)
--image.save(string.format('./res.jpg',config.model),res)

print('| done')
collectgarbage()
